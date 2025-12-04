# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrax LoggingBackend implemenation for Tensorboard."""

import time
import jax
import numpy as np
from tensorboardX import writer

_DEFAULT_STEP = 0


def _get_step(kwargs: dict[str, str | int]) -> int:
  """Returns the step from the kwargs, or 0 if not provided."""
  step = kwargs.get("step")
  return _DEFAULT_STEP if step is None else int(step)


def _preprocess_event_name(event_name: str) -> str:
  """Preprocesses the event name before logging."""
  return event_name.lstrip("/")  # Remove leading slashes


class TensorboardBackend:
  """A logging backend for Tensorboard that conforms to the LoggingBackend protocol."""

  def __init__(
      self,
      log_dir: str,
      flush_every_n_steps: int = 100,
      flush_interval_s: float = 30.0,
  ):
    self._flush_every_n_steps = flush_every_n_steps
    self._flush_interval_s = flush_interval_s
    self._last_flush_time = time.time()

    if jax.process_index() == 0:
      self._writer = writer.SummaryWriter(logdir=log_dir)
    else:
      self._writer = None

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    if self._writer is None:
      return
    current_step = _get_step(kwargs)
    event_name = _preprocess_event_name(event)
    self._writer.add_scalar(event_name, value, current_step)
    if current_step % self._flush_every_n_steps == 0:
      now = time.time()
      if (now - self._last_flush_time) >= self._flush_interval_s:
        self._writer.flush()
        self._last_flush_time = now

  def close(self):
    if self._writer:
      self._writer.close()
