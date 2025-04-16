import jax.numpy as jnp
from src.metrax.accuracy import accuracy

def test_accuracy():
    y_true = jnp.array([0, 1, 1, 0])
    y_pred = jnp.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # Predicted probabilities
    assert accuracy(y_true, y_pred) == 0.5  # 2 out of 4 correct
