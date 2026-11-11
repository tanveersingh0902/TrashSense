# TRASHSENSE - Helper Utility Functions

# FILE LOCATION: utils/helpers.py
#
# PURPOSE: Contains reusable helper functions used across
#          the project — plotting, visualization, printing.
#
# WHY SEPARATE FILE: Instead of writing plotting code in
#          every script, we write it ONCE here and import it
#          wherever needed. This follows the DRY principle
#          (Don't Repeat Yourself)


import os
import random

# --- Matplotlib: For creating charts and graphs ---
import matplotlib.pyplot as plt

# --- NumPy: For working with arrays of numbers ---
import numpy as np

# --- Pillow (PIL): For opening and processing image files ---
from PIL import Image


def count_images_per_class(dataset_directory):
    """
    Count how many images are in each class (subfolder).

    In our dataset, each category has its own folder:
        data/raw/cardboard/  → contains cardboard images
        data/raw/glass/      → contains glass images
        ... and so on

    Parameters:
        dataset_directory (str): Path to the folder containing class subfolders.

    Returns:
        class_counts (dict): Dictionary like {"cardboard": 403, "glass": 501, ...}
    """
    class_counts = {}

    # os.listdir() returns all files and folders inside a directory
    # sorted() puts them in alphabetical order
    for class_name in sorted(os.listdir(dataset_directory)):
        class_folder = os.path.join(dataset_directory, class_name)

        # os.path.isdir() checks if the path is a folder (not a file)
        # We only want folders, not loose files
        if os.path.isdir(class_folder):
            # Count files that look like images (by their file extension)
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
            image_count = 0

            for filename in os.listdir(class_folder):
                # .lower() converts to lowercase so ".JPG" and ".jpg" both match
                if filename.lower().endswith(image_extensions):
                    image_count += 1

            class_counts[class_name] = image_count

    return class_counts


def plot_class_distribution(class_counts, title="Image Distribution per Class"):
    """
    Create a bar chart showing how many images each class has.

    WHY THIS MATTERS:
    If one class has 500 images and another has only 50, the model
    might become biased toward the larger class. This chart helps
    us spot such imbalances early.

    Parameters:
        class_counts (dict): Dictionary of {class_name: count}.
        title (str): Title for the chart.
    """
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)

    # Create a figure (the canvas) and axes (the chart area)
    figure, axes = plt.subplots(figsize=(10, 6))

    # Create colored bars — one for each class
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    bars = axes.bar(class_names, counts, color=bar_colors[:len(class_names)],
                    edgecolor="black", linewidth=0.5)

    # Add the count number on top of each bar
    for bar, count in zip(bars, counts):
        percentage = (count / total) * 100
        axes.text(
            bar.get_x() + bar.get_width() / 2,   # x position (center of bar)
            bar.get_height() + 5,                  # y position (just above bar)
            f"{count}\n({percentage:.1f}%)",        # text to display
            ha="center",                            # horizontally centered
            fontsize=10,
            fontweight="bold",
        )

    # Labels and title
    axes.set_xlabel("Waste Category", fontsize=12)
    axes.set_ylabel("Number of Images", fontsize=12)
    axes.set_title(f"{title}\nTotal: {total} images", fontsize=14, fontweight="bold")

    # plt.tight_layout() adjusts spacing so nothing gets cut off
    plt.tight_layout()

    return figure


def display_sample_images(dataset_directory, num_samples=3, save_path=None):
    """
    Display a grid of sample images from each class.

    This helps us visually verify:
    1. Images are loading correctly
    2. Each class contains the right type of images
    3. Images are the right size and quality

    Parameters:
        dataset_directory (str): Path to the folder containing class subfolders.
        num_samples (int): How many sample images to show per class.
        save_path (str): If provided, save the figure to this file path.

    Returns:
        figure: The matplotlib figure object.
    """
    class_names = sorted([
        name for name in os.listdir(dataset_directory)
        if os.path.isdir(os.path.join(dataset_directory, name))
    ])

    num_classes = len(class_names)

    # Create a grid of subplots: rows = number of classes, columns = number of samples
    figure, axes_grid = plt.subplots(
        num_classes, num_samples,
        figsize=(num_samples * 3, num_classes * 3)
    )

    for row_index, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_directory, class_name)

        # Get all image file names in this class folder
        image_files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Randomly pick a few samples
        # random.seed(42) makes the "random" selection the same every time
        random.seed(42)
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))

        for col_index, image_file in enumerate(selected_files):
            image_path = os.path.join(class_folder, image_file)

            # Open the image using Pillow
            image = Image.open(image_path)

            # Get the correct axes (subplot) for this position
            if num_classes == 1:
                current_axes = axes_grid[col_index]
            else:
                current_axes = axes_grid[row_index, col_index]

            # Display the image
            current_axes.imshow(image)

            # Show the class name only on the first column
            if col_index == 0:
                current_axes.set_ylabel(class_name.upper(), fontsize=11, fontweight="bold")

            # Hide the axis ticks (the small numbers on the edges)
            current_axes.set_xticks([])
            current_axes.set_yticks([])

            # Show image dimensions as the title
            width, height = image.size
            current_axes.set_title(f"{width}x{height}", fontsize=8)

    figure.suptitle("Sample Images from Each Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Sample images saved to: {save_path}")

    return figure


def print_split_summary(train_df, val_df, test_df):
    """
    Print a nicely formatted summary table of the train/val/test split.

    Parameters:
        train_df: DataFrame with training data info
        val_df:   DataFrame with validation data info
        test_df:  DataFrame with test data info
    """
    print()
    print("=" * 65)
    print("   DATASET SPLIT SUMMARY")
    print("=" * 65)

    total = len(train_df) + len(val_df) + len(test_df)

    print(f"\n  {'Split':<15} {'Images':>10} {'Percentage':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*12}")
    print(f"  {'Training':<15} {len(train_df):>10} {len(train_df)/total*100:>10.1f}%")
    print(f"  {'Validation':<15} {len(val_df):>10} {len(val_df)/total*100:>10.1f}%")
    print(f"  {'Testing':<15} {len(test_df):>10} {len(test_df)/total*100:>10.1f}%")
    print(f"  {'-'*15} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL':<15} {total:>10} {'100.0%':>12}")

    # Per-class breakdown
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Get all unique class names
    all_classes = sorted(train_df["label"].unique())

    for class_name in all_classes:
        train_count = len(train_df[train_df["label"] == class_name])
        val_count = len(val_df[val_df["label"] == class_name])
        test_count = len(test_df[test_df["label"] == class_name])
        class_total = train_count + val_count + test_count
        print(f"  {class_name:<15} {train_count:>8} {val_count:>8} {test_count:>8} {class_total:>8}")

    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<15} {len(train_df):>8} {len(val_df):>8} {len(test_df):>8} {total:>8}")
    print()
