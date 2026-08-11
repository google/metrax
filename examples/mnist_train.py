# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST example with JAX and Metrax."""

from typing import Any, Callable

from absl import app
from absl import flags
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import grain.python as grain
import jax
from jax import sharding
from jax.experimental import multihost_utils
import jax.numpy as jnp
import metrax
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh
PartitionSpec = sharding.PartitionSpec

_NUM_GRAIN_WORKERS = 8

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    128,
    'Per-device batch size for training and evaluation.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    0.005,
    'Learning rate for optimizer',
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    10,
    'Number of epochs to train for.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Working directory for checkpoints, metrics, and artifacts. Required.',
    required=True,
)

FLAGS = flags.FLAGS
TrainState = train_state.TrainState


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # Flatten
    x = nn.Dense(features=10)(x)  # Output for 10 classes
    return x


class MNISTPreprocessing(grain.MapTransform):
  """MNIST data preprocessing for JAX."""

  def map(self, features: dict[str, Any]) -> dict[str, Any]:
    # Ensure image is float32 and normalized
    image = features['image'].astype(np.float32) / 255.0
    # Add channel dimension if missing (MNIST is (28, 28) and needs (28, 28, 1))
    if len(image.shape) == 2:
      image = np.expand_dims(image, axis=-1)
    return {'image': image, 'label': features['label']}


def get_datasets(
    global_batch_size: int,
):
  """Loads and prepares MNIST train/test datasets using Grain."""
  num_hosts = jax.process_count()
  current_host_id = jax.process_index()
  per_host_batch_size = global_batch_size // num_hosts
  logging.info(
      'Host %d/%d: get_datasets loading data with per-host batch size %d '
      '(Global batch size: %d)',
      current_host_id,
      num_hosts,
      per_host_batch_size,
      global_batch_size,
  )

  mnist_builder = tfds.builder('mnist', file_format='array_record')
  if jax.process_index() == 0:
    mnist_builder.download_and_prepare()
  if hasattr(jax.distributed, 'barrier'):
    jax.distributed.barrier()

  # Read split info after dataset has been prepared
  num_train_examples = mnist_builder.info.splits['train'].num_examples
  num_test_examples = mnist_builder.info.splits['test'].num_examples

  ds_train_source = mnist_builder.as_data_source(split='train')
  ds_test_source = mnist_builder.as_data_source(split='test')

  train_dataloader = grain.load(
      source=ds_train_source,
      shuffle=True,
      seed=jax.process_index(),
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=[MNISTPreprocessing()],
      batch_size=global_batch_size,
      worker_count=_NUM_GRAIN_WORKERS,
      drop_remainder=True,
  )
  test_dataloader = grain.load(
      source=ds_test_source,
      shuffle=False,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=[MNISTPreprocessing()],
      batch_size=global_batch_size,
      worker_count=_NUM_GRAIN_WORKERS,
      drop_remainder=True,
  )

  return (
      train_dataloader,
      test_dataloader,
      num_train_examples,
      num_test_examples,
  )


def create_train_state(init_rng):
  """Creates initial `TrainState`."""
  cnn = CNN()
  dummy_input_shape = (_BATCH_SIZE.value, 28, 28, 1)
  params = cnn.init(init_rng, jnp.ones(dummy_input_shape))['params']
  tx = optax.sgd(_LEARNING_RATE.value)
  return TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def compute_metrics(logits: jax.Array, labels: jax.Array) -> dict[str, Any]:
  """Computes loss, accuracy, and Metrax accuracy metrics."""
  loss = jnp.mean(
      optax.softmax_cross_entropy(
          logits=logits, labels=jax.nn.one_hot(labels, num_classes=10)
      )
  )
  predictions = jnp.argmax(logits, -1)
  accuracy = jnp.mean(predictions == labels)

  # Metrax metric 1: Accuracy (using predictions vs labels)
  metrax_accuracy = metrax.Accuracy.from_model_output(predictions, labels)

  # Metrax metric 2: SparseCategoricalAccuracy (using logits vs labels)
  metrax_sparse_cat_accuracy = (
      metrax.SparseCategoricalAccuracy.from_model_output(logits, labels)
  )

  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'metrax_accuracy': metrax_accuracy,
      'metrax_sparse_cat_accuracy': metrax_sparse_cat_accuracy,
  }
  return metrics


def train_step(
    state: TrainState, batch: dict[str, jax.Array]
) -> tuple[TrainState, dict[str, Any]]:
  """Performs a single training step."""

  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss = jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes=10)
        )
    )
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)

  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, batch['label'])
  return state, metrics


def eval_step(state: TrainState, batch: dict[str, jax.Array]) -> dict[str, Any]:
  """Performs a single evaluation step."""
  logits = state.apply_fn({'params': state.params}, batch['image'])
  return compute_metrics(logits, batch['label'])


def log_and_accumulate_metrics_dict(
    current_host_id: int,
    step: int,
    total_steps: int,
    step_metrics: dict[str, Any],
    prefix: str,  # 'Train' or 'Eval'
    epoch_metrics: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
  """Logs per-step metrics and accumulates them into a dictionary for epoch summary."""
  logging.info(
      'Host %d: %s Step %d/%d, metrics: %s',
      current_host_id,
      prefix,
      step + 1,
      total_steps,
      step_metrics,
  )
  if (step + 1) % 100 == 0 and prefix == 'Train':
    logging.info(
        'Host %d: Epoch %d, Train Step %d/%d, Batch Loss: %.4f, '
        'Batch Accuracy: %.2f%%, Batch Metrax Accuracy: %.4f, '
        'Batch Metrax Sparse Categorical Accuracy: %.4f',
        current_host_id,
        epoch,
        step + 1,
        total_steps,
        step_metrics['loss'],
        step_metrics['accuracy'] * 100,
        step_metrics['metrax_accuracy'].compute(),
        step_metrics['metrax_sparse_cat_accuracy'].compute(),
    )
  # Accumulate metrics
  epoch_metrics['loss'] += step_metrics['loss']
  epoch_metrics['accuracy'] += step_metrics['accuracy']
  epoch_metrics['metrax_accuracy'] = epoch_metrics['metrax_accuracy'].merge(
      step_metrics['metrax_accuracy']
  )
  epoch_metrics['metrax_sparse_cat_accuracy'] = epoch_metrics[
      'metrax_sparse_cat_accuracy'
  ].merge(step_metrics['metrax_sparse_cat_accuracy'])

  return epoch_metrics


def run_training_epoch(
    epoch: int,
    state: TrainState,
    train_iter: Any,
    steps_per_epoch: int,
    jit_train_step: Callable[
        [TrainState, dict[str, jax.Array]], tuple[TrainState, dict[str, Any]]
    ],
    mesh: sharding.Mesh,
    batch_shardings_pytree: dict[str, sharding.NamedSharding],
    current_host_id: int,
) -> tuple[TrainState, dict[str, Any]]:
  """Runs a single training epoch."""
  train_metrics_epoch = {
      'loss': 0.0,
      'accuracy': 0.0,
      'metrax_accuracy': metrax.Accuracy.empty(),
      'metrax_sparse_cat_accuracy': metrax.SparseCategoricalAccuracy.empty(),
  }
  logging.info('Host %d: Starting Training Epoch %d...', current_host_id, epoch)
  for step in range(steps_per_epoch):
    host_batch = next(train_iter)
    batch_pspecs_pytree = {k: v.spec for k, v in batch_shardings_pytree.items()}
    sharded_batch = multihost_utils.host_local_array_to_global_array(
        host_batch, mesh, batch_pspecs_pytree
    )

    state, metrics = jit_train_step(state, sharded_batch)

    train_metrics_epoch = log_and_accumulate_metrics_dict(
        current_host_id,
        step,
        steps_per_epoch,
        metrics,
        'Train',
        train_metrics_epoch,
        epoch=epoch,
    )
  return state, train_metrics_epoch


def run_evaluation_epoch(
    epoch: int,
    state: TrainState,
    test_dataloader: Any,
    steps_per_test: int,
    jit_eval_step: Callable[[TrainState, dict[str, jax.Array]], dict[str, Any]],
    mesh: sharding.Mesh,
    batch_shardings_pytree: dict[str, sharding.NamedSharding],
    current_host_id: int,
) -> dict[str, Any]:
  """Runs a single evaluation epoch."""
  test_metrics_epoch = {
      'loss': 0.0,
      'accuracy': 0.0,
      'metrax_accuracy': metrax.Accuracy.empty(),
      'metrax_sparse_cat_accuracy': metrax.SparseCategoricalAccuracy.empty(),
  }
  logging.info(
      'Host %d: Starting Evaluation for Epoch %d...', current_host_id, epoch
  )
  # Create a new iterator for each evaluation pass
  test_iter = iter(test_dataloader)
  for step in range(steps_per_test):
    host_test_batch = next(test_iter)
    batch_pspecs_pytree = {k: v.spec for k, v in batch_shardings_pytree.items()}
    sharded_test_batch = multihost_utils.host_local_array_to_global_array(
        host_test_batch, mesh, batch_pspecs_pytree
    )

    metrics = jit_eval_step(state, sharded_test_batch)

    test_metrics_epoch = log_and_accumulate_metrics_dict(
        current_host_id,
        step,
        steps_per_test,
        metrics,
        'Eval',
        test_metrics_epoch,
        epoch=epoch,
    )
  return test_metrics_epoch


def train_and_evaluate() -> TrainState:
  """Executes model training and evaluation loop with sharded data."""
  num_devices = jax.device_count()
  num_hosts = jax.process_count()
  current_host_id = jax.process_index()
  local_device_count = jax.local_device_count()

  per_device_batch_size = _BATCH_SIZE.value
  global_batch_size = num_devices * per_device_batch_size
  per_host_batch_size = global_batch_size // num_hosts

  logging.info(
      'Host %d/%d: Global Batch=%d, Per-Host Batch=%d, Per-Device Batch=%d, '
      'Num Devices Globally=%d, Local Devices=%d',
      current_host_id,
      num_hosts,
      global_batch_size,
      per_host_batch_size,
      per_device_batch_size,
      num_devices,
      local_device_count,
  )

  summary_writer = None
  if jax.process_index() == 0:  # Only host 0 writes summaries
    summary_writer = tensorboard.SummaryWriter(_WORKDIR.value)
    logging.info(
        'Host 0: Tensorboard summary writer initialized at %s', _WORKDIR.value
    )

  train_dataloader, test_dataloader, num_train_examples, num_test_examples = (
      get_datasets(global_batch_size)
  )
  train_iter = iter(train_dataloader)

  devices = list(jax.devices())
  devices.sort(key=lambda d: (d.process_index, d.id))
  mesh = Mesh(np.array(devices), ('data',))
  logging.info(
      'Host %d: Mesh defined: %s, local devices in mesh: %s',
      current_host_id,
      mesh,
      mesh.local_devices,
  )

  # Images (B, H, W, C), spec: (P('data'), P(None), P(None), P(None))
  # Labels (B,), spec: (P('data'),)
  image_sharding = NamedSharding(mesh, PartitionSpec('data', None, None, None))
  label_sharding = NamedSharding(mesh, PartitionSpec('data'))
  batch_shardings_pytree = {'image': image_sharding, 'label': label_sharding}
  state_sharding = NamedSharding(mesh, PartitionSpec())

  rng = jax.random.key(0)
  rng = jax.random.fold_in(rng, jax.process_index())
  _, init_rng = jax.random.split(rng)

  state = jax.jit(create_train_state, out_shardings=state_sharding)(init_rng)
  jit_train_step = jax.jit(
      train_step,
      in_shardings=(state_sharding, batch_shardings_pytree),
  )
  jit_eval_step = jax.jit(
      eval_step,
      in_shardings=(state_sharding, batch_shardings_pytree),
  )

  steps_per_epoch = num_train_examples // global_batch_size
  steps_per_test = num_test_examples // global_batch_size
  logging.info(
      'Host %d: Steps per epoch: %d, Steps per test: %d',
      current_host_id,
      steps_per_epoch,
      steps_per_test,
  )

  for epoch in range(1, _NUM_EPOCHS.value + 1):
    # Training phase
    state, train_metrics_epoch = run_training_epoch(
        epoch,
        state,
        train_iter,
        steps_per_epoch,
        jit_train_step,
        mesh,
        batch_shardings_pytree,
        current_host_id,
    )
    avg_train_loss = train_metrics_epoch['loss'] / steps_per_epoch
    avg_train_accuracy = train_metrics_epoch['accuracy'] / steps_per_epoch
    avg_train_metrax_accuracy = train_metrics_epoch['metrax_accuracy'].compute()
    avg_train_metrax_sparse_cat_acc = train_metrics_epoch[
        'metrax_sparse_cat_accuracy'
    ].compute()

    # Evaluation phase
    test_metrics_epoch = run_evaluation_epoch(
        epoch,
        state,
        test_dataloader,
        steps_per_test,
        jit_eval_step,
        mesh,
        batch_shardings_pytree,
        current_host_id,
    )
    avg_test_loss = test_metrics_epoch['loss'] / steps_per_test
    avg_test_accuracy = test_metrics_epoch['accuracy'] / steps_per_test
    avg_test_metrax_accuracy = test_metrics_epoch['metrax_accuracy'].compute()
    avg_test_metrax_sparse_cat_acc = test_metrics_epoch[
        'metrax_sparse_cat_accuracy'
    ].compute()

    logging.info(
        'Host %d: Epoch %d Summary -\n'
        '  Train Loss: %.4f, Train Accuracy: %.2f%%, '
        'Train Metrax Accuracy: %.4f, '
        'Train Metrax Sparse Categorical Accuracy: %.4f\n'
        '  Test Loss:  %.4f, Test Accuracy:  %.2f%%, '
        'Test Metrax Accuracy:  %.4f, '
        'Test Metrax Sparse Categorical Accuracy:  %.4f',
        current_host_id,
        epoch,
        avg_train_loss,
        avg_train_accuracy * 100,
        avg_train_metrax_accuracy,
        avg_train_metrax_sparse_cat_acc,
        avg_test_loss,
        avg_test_accuracy * 100,
        avg_test_metrax_accuracy,
        avg_test_metrax_sparse_cat_acc,
    )

    if jax.process_index() == 0 and summary_writer:
      summary_writer.scalar('train_loss_epoch', avg_train_loss, epoch)
      summary_writer.scalar('train_accuracy_epoch', avg_train_accuracy, epoch)
      summary_writer.scalar(
          'train_metrax_accuracy_epoch', avg_train_metrax_accuracy, epoch
      )
      summary_writer.scalar(
          'train_metrax_sparse_cat_accuracy_epoch',
          avg_train_metrax_sparse_cat_acc,
          epoch,
      )
      summary_writer.scalar('test_loss_epoch', avg_test_loss, epoch)
      summary_writer.scalar('test_accuracy_epoch', avg_test_accuracy, epoch)
      summary_writer.scalar(
          'test_metrax_accuracy_epoch', avg_test_metrax_accuracy, epoch
      )
      summary_writer.scalar(
          'test_metrax_sparse_cat_accuracy_epoch',
          avg_test_metrax_sparse_cat_acc,
          epoch,
      )
      summary_writer.flush()

  if jax.process_index() == 0 and summary_writer:
    summary_writer.close()

  if hasattr(jax.distributed, 'barrier'):
    jax.distributed.barrier()

  return state


def main(_):
  # Initialize standard distributed JAX backend (no-op on single-host CPU/GPU).
  try:
    jax.distributed.initialize()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info('Distributed JAX initialization skipped: %s', e)

  # Ensure TF does not allocate GPU memory if JAX is using it.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX Process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX Local Devices: %s', jax.local_devices())
  logging.info('JAX Global Devices: %s', jax.devices())

  if _WORKDIR.value is None:
    raise app.UsageError('The --workdir flag must be set.')

  train_and_evaluate()
  logging.info(
      'Host %d: Training and evaluation complete.', jax.process_index()
  )


if __name__ == '__main__':
  app.run(main)
