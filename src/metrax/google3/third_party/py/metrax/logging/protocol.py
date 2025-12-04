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

"""Metrax LoggingBackend protocol."""

from typing import Protocol
import numpy as np


class LoggingBackend(Protocol):
  """Defines the interface for a pluggable logging backend."""

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    """Logs a scalar value. Must match jax.monitoring listener signature."""
    ...

  def close(self):
    """Closes the logger and flushes any pending data."""
    ...
