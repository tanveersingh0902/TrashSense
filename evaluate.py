"""Evaluate the trained TrashSense model on Stage 2's untouched test split.

This script never calls model.fit(), so it never trains or changes the saved
model weights.  It uses only data/processed/test_split.csv, which Stage 2
reserved for this final evaluation.

Run from the project root with:
    venv\Scripts\python.exe evaluate.py
"""

import json
import os

import matplotlib

# Save charts to files without requiring a graph window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


# Inputs from earlier stages.
TEST_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, "test_split.csv")
CLASS_MAPPING_PATH = os.path.join(config.PROCESSED_DATA_DIR, "class_mapping.csv")
MODEL_PATH = config.MODEL_SAVE_PATH

# All Stage 5 results are kept together in one folder.
EVALUATION_DIR = os.path.join(config.OUTPUTS_DIR, "evaluation")
SUMMARY_PATH = os.path.join(EVALUATION_DIR, "test_summary.json")
REPORT_TEXT_PATH = os.path.join(EVALUATION_DIR, "classification_report.txt")
REPORT_CSV_PATH = os.path.join(EVALUATION_DIR, "classification_report.csv")
CONFUSION_MATRIX_CSV_PATH = os.path.join(EVALUATION_DIR, "confusion_matrix.csv")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(EVALUATION_DIR, "confusion_matrix.png")
PREDICTIONS_PATH = os.path.join(EVALUATION_DIR, "test_predictions.csv")


def load_test_data():
    """Load Stage 2's test list and its saved class-number mapping."""
    for required_path in (TEST_CSV_PATH, CLASS_MAPPING_PATH, MODEL_PATH):
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"Required file was not found: {required_path}\n"
                "Stage 5 needs the Stage 2 test split and the best Stage 4 model."
            )

    test_df = pd.read_csv(TEST_CSV_PATH)
    mapping_df = pd.read_csv(CLASS_MAPPING_PATH)

    if not {"filepath", "label"}.issubset(test_df.columns):
        raise ValueError("test_split.csv must contain 'filepath' and 'label' columns.")
    if not {"class_name", "class_index"}.issubset(mapping_df.columns):
        raise ValueError("class_mapping.csv must contain 'class_name' and 'class_index' columns.")

    mapping_df = mapping_df.sort_values("class_index")
    class_names = mapping_df["class_name"].tolist()
    expected_indices = list(range(config.NUM_CLASSES))
    if mapping_df["class_index"].tolist() != expected_indices:
        raise ValueError(f"Expected class indices {expected_indices} in class_mapping.csv.")
    if class_names != config.WASTE_CATEGORIES:
        raise ValueError(
            "The Stage 2 class mapping does not match config.WASTE_CATEGORIES. "
            f"Found: {class_names}"
        )
    if set(test_df["label"]) != set(class_names):
        raise ValueError("The test labels do not match the saved class mapping.")

    missing_images = test_df.loc[~test_df["filepath"].map(os.path.exists), "filepath"]
    if not missing_images.empty:
        raise FileNotFoundError(
            "Some test images listed by Stage 2 no longer exist. "
            "Re-run Stage 2 before evaluating."
        )

    return test_df, class_names


def create_test_generator(test_df, class_names):
    """Load the untouched test images with MobileNetV2 preprocessing only."""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        classes=class_names,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )


def save_confusion_matrix_plot(matrix, class_names):
    """Create and save a labelled visual confusion matrix."""
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of test images")

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="TrashSense Test-Set Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    midpoint = matrix.max() / 2 if matrix.max() else 0
    for row, column in np.ndindex(matrix.shape):
        color = "white" if matrix[row, column] > midpoint else "black"
        axis.text(column, row, matrix[row, column], ha="center", va="center", color=color)

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=150)
    plt.close(figure)


def main():
    """Load the trained model, evaluate it once, and save all test results."""
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    test_df, class_names = load_test_data()
    test_generator = create_test_generator(test_df, class_names)

    # compile=False ensures loading does not restore an optimizer for training.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.input_shape[1:] != config.INPUT_SHAPE:
        raise ValueError(
            f"Model input shape {model.input_shape[1:]} does not match "
            f"the configured image shape {config.INPUT_SHAPE}."
        )
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs but the dataset has "
            f"{len(class_names)} classes."
        )
    # Compilation here only tells Keras how to calculate test loss and accuracy.
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    # With shuffle=False, prediction row i matches actual-label row i.
    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=1)
    predicted_indices = np.argmax(probabilities, axis=1)
    actual_indices = test_generator.classes

    report_text = classification_report(
        actual_indices,
        predicted_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dictionary = classification_report(
        actual_indices,
        predicted_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual_indices, predicted_indices)

    with open(REPORT_TEXT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    pd.DataFrame(report_dictionary).transpose().to_csv(REPORT_CSV_PATH)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        CONFUSION_MATRIX_CSV_PATH
    )
    save_confusion_matrix_plot(matrix, class_names)

    prediction_df = test_df.copy()
    prediction_df["actual_index"] = actual_indices
    prediction_df["predicted_index"] = predicted_indices
    prediction_df["predicted_label"] = [class_names[index] for index in predicted_indices]
    prediction_df["confidence"] = np.max(probabilities, axis=1)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    summary = {
        "test_images": int(test_generator.samples),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\nStage 5 evaluation complete. The model was not trained or modified.")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Results folder: {EVALUATION_DIR}")
    print(f"Classification report: {REPORT_TEXT_PATH}")
    print(f"Confusion-matrix plot: {CONFUSION_MATRIX_PLOT_PATH}")


if __name__ == "__main__":
    main()
