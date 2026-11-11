"""Build the TrashSense model and check its input and output shapes."""

import tensorflow as tf

import config
from models.model import build_model


def main():
    """Create the model and verify its expected shapes without training."""
    model = build_model()
    model.summary()

    expected_input_shape = (None, *config.INPUT_SHAPE)
    expected_output_shape = (None, config.NUM_CLASSES)

    assert model.input_shape == expected_input_shape, (
        f"Expected input shape {expected_input_shape}, got {model.input_shape}"
    )
    assert model.output_shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, got {model.output_shape}"
    )

    # A batch dimension is required, even when testing only one image.
    sample_batch = tf.zeros((1, *config.INPUT_SHAPE), dtype=tf.float32)
    class_scores = model(sample_batch, training=False)
    expected_scores_shape = (1, config.NUM_CLASSES)

    assert tuple(class_scores.shape) == expected_scores_shape, (
        f"Expected {config.NUM_CLASSES} class scores, got shape {class_scores.shape}"
    )
    tf.debugging.assert_near(
        tf.reduce_sum(class_scores, axis=1),
        tf.ones((1,), dtype=tf.float32),
        message="Softmax class scores should add up to 1.",
    )

    print(f"\nInput shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Forward-pass class-score shape: {tuple(class_scores.shape)}")
    print("Model shape checks passed. No training was performed.")


if __name__ == "__main__":
    main()
