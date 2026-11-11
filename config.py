# ============================================================
# TRASHSENSE 2.0 - Project Configuration
# ============================================================
# PURPOSE: This file stores all the settings/constants used
#          across the entire project in ONE place.
#
# WHY: Instead of writing numbers like 224 or 32 in every file,
#      we define them here ONCE. If we need to change a value,
#      we change it in ONE place instead of hunting through
#      every file.
# ============================================================

import os

# --- Project Root Directory ---
# os.path.dirname() gets the folder that contains this file.
# os.path.abspath() converts it to a full/absolute path.
# This gives us the path to the TrashSense folder no matter
# where we run the code from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# --- Folder Paths ---
# os.path.join() combines folder names in a way that works
# on both Windows (uses \) and Mac/Linux (uses /).

DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR      = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR        = os.path.join(PROJECT_ROOT, "models")
SAVED_MODELS_DIR  = os.path.join(MODELS_DIR, "saved_models")

OUTPUTS_DIR       = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR         = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR       = os.path.join(OUTPUTS_DIR, "reports")


# --- Image Settings ---
# All images will be resized to this size before feeding into the CNN.
# MobileNetV2 expects 224x224 pixel images.
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
IMAGE_SIZE   = (IMAGE_HEIGHT, IMAGE_WIDTH)   # Tuple format: (224, 224)

# Images have 3 color channels: Red, Green, Blue
IMAGE_CHANNELS = 3
INPUT_SHAPE    = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)  # (224, 224, 3)


# --- Training Settings ---
# BATCH_SIZE: How many images the model looks at in one step.
#   - Smaller batch = less memory needed, but slower training.
#   - Larger batch  = more memory needed, but faster training.
#   - 32 is a good default for most computers.
BATCH_SIZE = 32

# EPOCHS: How many times the model sees the ENTIRE dataset.
#   - More epochs = model learns more (but can overfit).
#   - Fewer epochs = model learns less (but trains faster).
#   - We start with 20 and can adjust later.
EPOCHS = 20

# LEARNING_RATE: How big of a step the model takes when learning.
#   - Too high = model jumps around and never settles on good answers.
#   - Too low  = model learns very slowly.
#   - 0.0001 is a safe starting point for transfer learning.
LEARNING_RATE = 0.0001

# VALIDATION_SPLIT: What fraction of training data to use for validation.
#   - 0.2 means 20% of data is used to check performance during training.
#   - The remaining 80% is used for actual learning.
VALIDATION_SPLIT = 0.2


# --- Waste Categories ---
# These are the 6 types of waste our model will learn to classify.
# The dataset should have one folder per category inside data/raw/.
WASTE_CATEGORIES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Total number of categories (used when building the model's output layer).
NUM_CLASSES = len(WASTE_CATEGORIES)   # = 6


# --- Model File Names ---
# Where to save the trained model after training is complete.
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "trashsense_model.keras")
