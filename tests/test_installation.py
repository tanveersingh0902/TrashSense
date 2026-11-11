# TRASHSENSE - Installation Verification Test

# PURPOSE: Run this file AFTER installing requirements.txt
#          to check if all libraries are installed correctly.
#
# HOW TO RUN: python tests/test_installation.py


def check_library(library_name, import_name=None):
    """
    Try to import a library and print whether it was successful.
    
    Parameters:
        library_name (str): The display name of the library (for printing).
        import_name (str):  The actual Python import name (if different from display name).
                            For example, Pillow is installed as 'Pillow' but imported as 'PIL'.
    """
    # If no separate import name is given, use the display name
    if import_name is None:
        import_name = library_name

    try:
        # __import__() is a built-in Python function that imports a module by name (as a string).
        # It's the same as writing "import tensorflow" but lets us use a variable.
        module = __import__(import_name)

        # Most libraries have a __version__ attribute that tells us which version is installed.
        version = getattr(module, "__version__", "version not available")

        print(f"  ✅ {library_name:20s} → version {version}")
        return True

    except ImportError:
        # If the import fails, the library is NOT installed.
        print(f"  ❌ {library_name:20s} → NOT INSTALLED")
        return False


def main():
    """
    Main function that checks all required libraries one by one.
    """
    print("=" * 60)
    print("   TRASHSENSE - Installation Verification")
    print("=" * 60)
    print()

    # List of all libraries we need.
    # Format: (display_name, import_name)
    # Some libraries have different install names vs import names:
    #   - "Pillow" is installed via pip, but we import it as "PIL"
    #   - "scikit-learn" is installed via pip, but we import it as "sklearn"
    libraries_to_check = [
        ("TensorFlow",    "tensorflow"),
        ("NumPy",         "numpy"),
        ("Pandas",        "pandas"),
        ("Pillow (PIL)",  "PIL"),
        ("Scikit-learn",  "sklearn"),
        ("Matplotlib",    "matplotlib"),
        ("Seaborn",       "seaborn"),
        ("Streamlit",     "streamlit"),
        ("tqdm",          "tqdm"),
    ]

    print("Checking installed libraries:\n")

    # Track how many passed and failed
    passed = 0
    failed = 0

    for display_name, import_name in libraries_to_check:
        if check_library(display_name, import_name):
            passed += 1
        else:
            failed += 1

    # --- Summary ---
    print()
    print("-" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("-" * 60)

    if failed == 0:
        print()
        print("  🎉 ALL LIBRARIES INSTALLED SUCCESSFULLY!")
        print("  You are ready to build TrashSense!")
    else:
        print()
        print("  ⚠️  Some libraries are missing.")
        print("  Run this command to install them:")
        print("  → pip install -r requirements.txt")

    print()

    # --- Bonus: Check if GPU is available for TensorFlow ---
    print("=" * 60)
    print("   GPU Check (Optional - Training is faster with GPU)")
    print("=" * 60)
    print()

    try:
        import tensorflow as tf

        # List all GPUs that TensorFlow can see
        gpus = tf.config.list_physical_devices("GPU")

        if len(gpus) > 0:
            print(f"  🚀 GPU AVAILABLE! Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"     - {gpu.name}")
        else:
            print("  💻 No GPU found. Training will use CPU.")
            print("     (This is fine — it will just be slower.)")

    except ImportError:
        print("  ⚠️  TensorFlow not installed — cannot check GPU.")

    print()


# This block runs when you execute the file directly: python test_installation.py
# It does NOT run when another file imports this file.
if __name__ == "__main__":
    main()
