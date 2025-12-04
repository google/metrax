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

"""Metrax LoggingBackend implemenation for Weight & Bias."""

import datetime
import logging

import jax
import numpy as np

_DEFAULT_STEP = 0


def _get_step(kwargs: dict[str, str | int]) -> int:
  """Returns the step from the kwargs, or 0 if not provided."""
  step = kwargs.get("step")
  return _DEFAULT_STEP if step is None else int(step)


def _preprocess_event_name(event_name: str) -> str:
  """Preprocesses the event name before logging."""
  return event_name.lstrip("/")  # Remove leading slashes


class WandbBackend:
  """A logging backend for W&B that conforms to the LoggingBackend protocol."""

  def __init__(self, project: str, name: str | None = None, **kwargs):
    if jax.process_index() != 0:
      self._is_active = False
      self.wandb = None  # Ensure the attribute exists
      return

    try:
      # pylint: disable=g-import-not-at-top
      # pytype: disable=import-error
      import wandb
    except ImportError as e:
      raise ImportError(
          "The 'wandb' library is not installed. Please install it with "
          "'pip install wandb' to use the WandbBackend."
      ) from e
    self.wandb = wandb

    run_name = name or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project=project, name=run_name, anonymous="allow", **kwargs)
    if wandb.run:
      logging.info("W&B run URL: %s", wandb.run.url)
      self._is_active = True
    else:
      self._is_active = False

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    if self.wandb is None or not self._is_active:
      return
    current_step = _get_step(kwargs)
    event_name = _preprocess_event_name(event)
    self.wandb.log({event_name: value}, step=current_step)

  def close(self):
    if self.wandb is None or not self._is_active:
      return
    if hasattr(self.wandb, "run") and self.wandb.run:
      self.wandb.finish()
