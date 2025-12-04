"""Tests for the W&B backend."""

import builtins
from unittest import mock

from absl.testing import absltest
import metrax.logging as metrax_logging

WandbBackend = metrax_logging.WandbBackend


_real_import = builtins.__import__


class WandbBackendTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_wandb = mock.Mock()
    self.mock_wandb.run = mock.Mock()
    self.mock_datetime = mock.Mock()
    self.mock_datetime.datetime.now.return_value.strftime.return_value = (
        "fixed-run-name"
    )

  def _mock_successful_import(self, name, *args, **kwargs):
    """Mock __import__ to return our mock_wandb for 'wandb'."""
    if name == "wandb":
      return self.mock_wandb
    return _real_import(name, *args, **kwargs)

  def test_init_and_log_success_main_process(self):
    """Tests successful init, logging, and closing on the main process."""
    with mock.patch("jax.process_index", return_value=0), mock.patch(
        "metrax.logging.wandb_backend.datetime", self.mock_datetime
    ), mock.patch(
        "builtins.__import__", side_effect=self._mock_successful_import
    ):

      backend = WandbBackend(project="test-project")
      self.mock_wandb.init.assert_called_once_with(
          project="test-project", name="fixed-run-name", anonymous="allow"
      )
      self.assertTrue(backend._is_active)

      backend.log_scalar("/myevent", 123.45, step=50)
      self.mock_wandb.log.assert_called_once_with({"myevent": 123.45}, step=50)

      backend.close()
      self.mock_wandb.finish.assert_called_once()

  def test_init_non_main_process_is_noop(self):
    """Tests that the backend does nothing on non-main processes."""
    with mock.patch("jax.process_index", return_value=1):
      backend = WandbBackend(project="test-project")
      self.assertFalse(backend._is_active)
      self.assertIsNone(backend.wandb)

  def test_init_fails_if_wandb_not_installed(self):
    """Tests that __init__ raises an ImportError if wandb is missing."""

    def failing_import(name, *args, **kwargs):
      """Mock __import__ to raise an error for 'wandb'."""
      if name == "wandb":
        raise ImportError("Mocked import failure")
      return _real_import(name, *args, **kwargs)

    with mock.patch(
        "builtins.__import__", side_effect=failing_import
    ), mock.patch("jax.process_index", return_value=0):
      with self.assertRaises(ImportError) as cm:
        WandbBackend(project="test-project")
      self.assertIn("pip install wandb", str(cm.exception))


if __name__ == "__main__":
  absltest.main()
