# Copyright 2025 Google LLC
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

import sys
from unittest import mock

from absl.testing import absltest
from metrax import logging as metrax_logging

WandbBackend = metrax_logging.WandbBackend


class WandbBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_wandb = mock.Mock()
    self.mock_wandb.run = mock.Mock()

    self.mock_datetime = mock.Mock()
    self.mock_datetime.datetime.now.return_value.strftime.return_value = (
        "run-name"
    )

  def test_init_and_log_success_main_process(self):
    """Tests successful init, logging, and closing on the main process."""
    with mock.patch("jax.process_index", return_value=0), mock.patch(
        "metrax.logging.wandb_backend.datetime", self.mock_datetime
    ), mock.patch.dict("sys.modules", {"wandb": self.mock_wandb}):

      backend = WandbBackend(project="test-project")
      self.mock_wandb.init.assert_called_once_with(
          project="test-project", name="run-name", anonymous="allow"
      )
      self.assertTrue(backend._is_active)

      backend.log_scalar("/myevent", 123.45, step=50)
      self.mock_wandb.log.assert_called_once_with({"myevent": 123.45}, step=50)

      backend.close()
      self.mock_wandb.finish.assert_called_once()

  def test_init_non_main_process_is_noop(self):
    """Tests that the backend does nothing on non-main processes."""
    with mock.patch("jax.process_index", return_value=1), mock.patch.dict(
        "sys.modules", {"wandb": self.mock_wandb}
    ):

      backend = WandbBackend(project="test-project")
      self.assertFalse(backend._is_active)
      self.mock_wandb.init.assert_not_called()

      backend.log_scalar("myevent", 1.0, step=1)
      self.mock_wandb.log.assert_not_called()

      backend.close()
      self.mock_wandb.finish.assert_not_called()

  def test_init_fails_if_wandb_not_installed(self):
    """Tests that __init__ raises an ImportError if wandb is missing."""
    with mock.patch.dict(sys.modules):
      if "wandb" in sys.modules:
        del sys.modules["wandb"]

      with self.assertRaises(ImportError) as cm:
        WandbBackend(project="test-project")
      self.assertIn("pip install wandb", str(cm.exception))


if __name__ == "__main__":
  absltest.main()
