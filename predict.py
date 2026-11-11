"""Predict one waste class from one image using the trained TrashSense model.

This is inference only: it loads the Stage 4 checkpoint and never calls
model.fit(), so it does not train or change the model.

Example:
    venv\Scripts\python.exe predict.py "data\raw\dataset-resized\plastic\plastic1.jpg"
    venv\Scripts\python.exe predict.py "my_photo.jpg" --threshold 0.70
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config


# A prediction below 60% is displayed as uncertain, not as a reliable answer.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
CLASS_MAPPING_PATH = os.path.join(config.PROCESSED_DATA_DIR, "class_mapping.csv")


def load_class_names():
    """Read Stage 2's class-to-index mapping in its saved index order."""
    if not os.path.exists(CLASS_MAPPING_PATH):
        raise FileNotFoundError(f"Class mapping file not found: {CLASS_MAPPING_PATH}")

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8", newline="") as mapping_file:
        rows = list(csv.DictReader(mapping_file))

    if not rows or set(rows[0]) != {"class_name", "class_index"}:
        raise ValueError("class_mapping.csv must contain class_name and class_index columns.")

    rows.sort(key=lambda row: int(row["class_index"]))
    class_names = [row["class_name"] for row in rows]
    class_indices = [int(row["class_index"]) for row in rows]

    if class_indices != list(range(config.NUM_CLASSES)):
        raise ValueError("class_mapping.csv must contain consecutive class indices from 0 to 5.")
    if class_names != config.WASTE_CATEGORIES:
        raise ValueError(
            "The saved class mapping does not match config.WASTE_CATEGORIES. "
            f"Found: {class_names}"
        )

    return class_names


def load_trained_model(class_names):
    """Load the saved Stage 4 checkpoint and check that it matches the dataset."""
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {config.MODEL_SAVE_PATH}\n"
            "Run Stage 4 training first, then use its best saved .keras model."
        )

    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH, compile=False)

    if model.input_shape[1:] != config.INPUT_SHAPE:
        raise ValueError(
            f"Model expects {model.input_shape[1:]}, but config.py specifies "
            f"{config.INPUT_SHAPE}."
        )
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, but there are "
            f"{len(class_names)} saved classes."
        )

    return model


def prepare_image(image_path):
    """Open one image and turn it into one MobileNetV2-ready image batch."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            resized_image = rgb_image.resize(config.IMAGE_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not open a valid image: {image_path}") from error

    preprocessed_image = preprocess_input(image_array)
    image_batch = np.expand_dims(preprocessed_image, axis=0)
    return image_batch


def predict_image(model, image_path, class_names):
    """Return the six scores, highest-scoring class, and its confidence."""
    image_batch = prepare_image(image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    return probabilities, predicted_label, confidence


def main():
    """Read command-line settings, make one prediction, and print the result."""
    parser = argparse.ArgumentParser(
        description="Classify one waste image with the trained TrashSense model."
    )
    parser.add_argument("image_path", help="Path to the image to classify.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence from 0 to 1 before the result is called reliable.",
    )
    arguments = parser.parse_args()

    if not 0.0 <= arguments.threshold <= 1.0:
        parser.error("--threshold must be a number from 0 to 1.")

    try:
        class_names = load_class_names()
        model = load_trained_model(class_names)
        probabilities, predicted_label, confidence = predict_image(
            model, arguments.image_path, class_names
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))

    print("\nClass probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name.capitalize():<10} {probability:.2%}")

    print(f"\nHighest-scoring class: {predicted_label.capitalize()}")
    print(f"Confidence: {confidence:.2%}")
    if confidence >= arguments.threshold:
        print("Result: Reliable enough to report at the selected threshold.")
    else:
        print(
            "Result: UNCERTAIN. The highest score is below the selected "
            f"{arguments.threshold:.0%} threshold; ask for a clearer image or manual review."
        )


if __name__ == "__main__":
    main()
