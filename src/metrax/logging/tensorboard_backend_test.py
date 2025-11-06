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

from unittest import mock

from absl.testing import absltest
import metrax.logging as metrax_logging

TensorboardBackend = metrax_logging.TensorboardBackend


class TensorboardBackendTest(absltest.TestCase):

  @mock.patch("metrax.logging.tensorboard_backend.writer.SummaryWriter")
  def test_init_and_log_success_main_process(self, mock_summary_writer):
    """Tests successful init, logging, and closing on the main process."""
    mock_writer_instance = mock_summary_writer.return_value

    with mock.patch("jax.process_index", return_value=0):
      backend = TensorboardBackend(log_dir="/fake/logs", flush_every_n_steps=2)

      mock_summary_writer.assert_called_once_with(logdir="/fake/logs")

      backend.log_scalar("/event1", 1.0, step=1)
      mock_writer_instance.add_scalar.assert_called_with("event1", 1.0, 1)
      mock_writer_instance.flush.assert_not_called()

      backend.log_scalar("event2", 2.0, step=2)
      mock_writer_instance.add_scalar.assert_called_with("event2", 2.0, 2)
      mock_writer_instance.flush.assert_called_once()

      backend.close()
      mock_writer_instance.close.assert_called_once()

  @mock.patch("metrax.logging.tensorboard_backend.writer.SummaryWriter")
  def test_init_non_main_process_is_noop(self, mock_summary_writer):
    """Tests that the backend does nothing on non-main processes."""
    mock_writer_instance = mock_summary_writer.return_value

    with mock.patch("jax.process_index", return_value=1):
      backend = TensorboardBackend(log_dir="/fake/logs")

      mock_summary_writer.assert_not_called()

      backend.log_scalar("myevent", 1.0, step=1)
      mock_writer_instance.add_scalar.assert_not_called()

      backend.close()
      mock_writer_instance.close.assert_not_called()


if __name__ == "__main__":
  absltest.main()
