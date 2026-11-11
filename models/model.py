"""
Build the TrashSense image-classification model.

The images passed to this model should already be prepared with
MobileNetV2's preprocess_input function, as in src/data_preprocessing.py.
"""

import tensorflow as tf

import config


def build_model():
    """Create and return a MobileNetV2 model for the six waste classes."""
    # Load MobileNetV2's image-recognition feature extractor.
    # include_top=False removes its original ImageNet classification head.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Keep the pretrained features unchanged while we first train our own head.
    base_model.trainable = False

    # Input shape is one image: height, width, and RGB color channels.
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="image")

    # Extract image features. training=False keeps pretrained BatchNorm layers
    # in inference mode, which is important when using a frozen base model.
    features = base_model(inputs, training=False)

    # Replace the original ImageNet head with a small TrashSense classifier.
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(
        features
    )
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        name="class_probabilities",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="trashsense_mobilenetv2")
