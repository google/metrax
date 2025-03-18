import jax.numpy as jnp

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (jnp.ndarray): True labels.
        y_pred (jnp.ndarray): Predicted labels.

    Returns:
        float: Accuracy as a fraction of correct predictions.
    """
    y_pred = jnp.argmax(y_pred, axis=-1)  # Get the predicted class
    correct_predictions = jnp.sum(y_true == y_pred)  # Count correct predictions
    total_predictions = y_true.shape[0]  # Total number of predictions
    return correct_predictions / total_predictions  # Calculate accuracy
