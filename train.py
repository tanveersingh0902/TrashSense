"""Train the TrashSense MobileNetV2 waste-classification model.

This script deliberately uses only the train and validation splits created in
Stage 2.  The test split stays untouched until Stage 5.

Run from the project root with:
    venv\Scripts\python.exe train.py
"""

import json
import os

import matplotlib

# "Agg" lets Matplotlib save image files without opening a graph window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
from models.model import build_model


# Files produced by Stage 2 and files produced by this training script.
TRAIN_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, "train_split.csv")
VALIDATION_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, "val_split.csv")
CLASS_MAPPING_PATH = os.path.join(config.PROCESSED_DATA_DIR, "class_mapping.csv")
HISTORY_PATH = os.path.join(config.REPORTS_DIR, "training_history.json")
ACCURACY_PLOT_PATH = os.path.join(config.PLOTS_DIR, "04_training_accuracy.png")
LOSS_PLOT_PATH = os.path.join(config.PLOTS_DIR, "05_training_loss.png")


def load_split_data():
    """Load and validate Stage 2's train and validation CSV files.

    The CSV files contain two columns: ``filepath`` and ``label``.  They are
    the source of truth for which images may be used in this stage.
    """
    for csv_path in (TRAIN_CSV_PATH, VALIDATION_CSV_PATH, CLASS_MAPPING_PATH):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Required Stage 2 file was not found: {csv_path}\n"
                "Run src/data_preprocessing.py before running train.py."
            )

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    validation_df = pd.read_csv(VALIDATION_CSV_PATH)
    class_mapping_df = pd.read_csv(CLASS_MAPPING_PATH)

    required_split_columns = {"filepath", "label"}
    required_mapping_columns = {"class_name", "class_index"}
    if not required_split_columns.issubset(train_df.columns) or not required_split_columns.issubset(validation_df.columns):
        raise ValueError("The train/validation CSV files must contain 'filepath' and 'label' columns.")
    if not required_mapping_columns.issubset(class_mapping_df.columns):
        raise ValueError("class_mapping.csv must contain 'class_name' and 'class_index' columns.")

    # Check the saved class mapping agrees with config.py and with both splits.
    expected_classes = set(config.WASTE_CATEGORIES)
    saved_classes = set(class_mapping_df["class_name"])
    split_classes = set(train_df["label"]).union(validation_df["label"])
    if saved_classes != expected_classes or split_classes != expected_classes:
        raise ValueError(
            "The Stage 2 class mapping or labels do not match config.WASTE_CATEGORIES. "
            f"Expected: {sorted(expected_classes)}"
        )

    missing_train_images = train_df.loc[~train_df["filepath"].map(os.path.exists), "filepath"]
    missing_validation_images = validation_df.loc[
        ~validation_df["filepath"].map(os.path.exists), "filepath"
    ]
    if not missing_train_images.empty or not missing_validation_images.empty:
        raise FileNotFoundError(
            "Some image paths saved in Stage 2 no longer exist. "
            "Re-run Stage 2 to recreate the split files."
        )

    return train_df, validation_df


def create_training_generators(train_df, validation_df):
    """Create Stage-2-compatible, categorical train and validation generators."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Stage 2 used class_mode="categorical", so each label is one-hot encoded.
    # The explicit class order preserves Stage 2's mapping: cardboard=0 ... trash=5.
    class_names = config.WASTE_CATEGORIES
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        classes=class_names,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col="filepath",
        y_col="label",
        classes=class_names,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator


def save_history(history):
    """Save Keras's per-epoch metrics as readable JSON."""
    serializable_history = {
        metric_name: [float(value) for value in values]
        for metric_name, values in history.history.items()
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as history_file:
        json.dump(serializable_history, history_file, indent=2)


def plot_history(history):
    """Save separate accuracy and loss charts for training and validation."""
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ran, history.history["accuracy"], label="Training accuracy")
    plt.plot(epochs_ran, history.history["val_accuracy"], label="Validation accuracy")
    plt.title("TrashSense Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ran, history.history["loss"], label="Training loss")
    plt.plot(epochs_ran, history.history["val_loss"], label="Validation loss")
    plt.title("TrashSense Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical cross-entropy loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()


def main():
    """Run Stage 4: load data, compile, train, and save learning evidence."""
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    train_df, validation_df = load_split_data()
    train_generator, validation_generator = create_training_generators(
        train_df, validation_df
    )

    # build_model() keeps the Stage 3 architecture unchanged.
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStage 4 is using only the saved training and validation splits.")
    print(f"Training images: {train_generator.samples}")
    print(f"Validation images: {validation_generator.samples}")
    print("Test images: not loaded; reserved for Stage 5.")
    print(f"Class mapping: {train_generator.class_indices}")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    save_history(history)
    plot_history(history)

    best_epoch = (
        min(
            range(len(history.history["val_loss"])),
            key=history.history["val_loss"].__getitem__,
        )
        + 1
    )
    print("\nTraining complete.")
    print(f"Best validation-loss epoch: {best_epoch}")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print(f"Accuracy chart saved to: {ACCURACY_PLOT_PATH}")
    print(f"Loss chart saved to: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
