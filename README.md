# TrashSense

TrashSense is a local Streamlit application that uses image classification to
suggest one of six waste categories: cardboard, glass, metal, paper, plastic,
or trash.

## Problem and objectives

Waste items can be difficult to sort by appearance. This project explores an
image classifier as decision support: estimate an item's category, show
general disposal guidance, and let a user review the result.

The project objectives are to:

- Classify an uploaded image into one of the six trained categories.
- Display the model's class scores and communicate uncertainty.
- Provide general disposal suggestions that defer to local rules.
- Keep session history and feedback available during the current app session.

## Features

- JPG and PNG image upload with RGB conversion and resizing.
- Six-class prediction and confidence-band display.
- A separate, heuristic ImageNet label screen for some clearly unrelated
  subjects. It is not a general waste detector.
- Session-only prediction history, feedback, dashboard, and CSV download.
- General disposal guidance for each supported category.

## Technologies

Python, TensorFlow/Keras, MobileNetV2, Streamlit, NumPy, Pandas, Pillow,
scikit-learn, and Matplotlib.

## Model architecture

The waste classifier uses ImageNet-pretrained MobileNetV2 as a frozen feature
extractor, followed by global average pooling, a 128-unit ReLU dense layer,
30% dropout, and a six-unit Softmax output. Images are converted to RGB,
resized to 224 × 224, and processed with MobileNetV2's `preprocess_input`
function. The training and inference preprocessing are intended to match.

## Dataset

The local resized dataset is organized by class under
`data/raw/dataset-resized/`. The current copy contains 2,527 images:

| Class | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |

The class counts are imbalanced. The raw images are ignored by Git and are
needed for retraining, not for loading the saved classifier. Processed split
CSV files and `class_mapping.csv` are in `data/processed/`. The split files
currently contain paths from this local checkout; regenerate them with the
preprocessing script if the project or dataset is moved.

## Project structure

```text
TrashSense/
├── .streamlit/
│   └── config.toml                 # Streamlit theme
├── data/
│   ├── processed/                  # Class mapping and train/validation/test lists
│   └── raw/dataset-resized/        # Local dataset; ignored by Git
├── models/
│   ├── model.py                    # Model architecture for training
│   └── saved_models/
│       └── trashsense_model.keras  # Trained model used for inference
├── outputs/                        # Local generated plots and training history; ignored
├── src/
│   └── data_preprocessing.py       # Dataset inspection and split preparation
├── tests/                          # Image-screen, model-shape, installation checks
├── utils/                          # Image screening, helpers, disposal guidance
├── app.py                          # Streamlit application
├── config.py                       # Shared paths, labels, and settings
├── evaluate.py                     # Test-split evaluation script
├── predict.py                      # Model loading and image prediction
├── train.py                        # Training script
├── requirements.txt
└── README.md
```

`data/raw/` and `outputs/` are excluded from Git by `.gitignore`. The model
checkpoint and processed split files are tracked in the repository.

## Install and run locally

From the project root, create and activate a virtual environment, then install
the listed dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the application locally with:

```powershell
streamlit run app.py
```

## How to use the application

1. Choose **Waste Detection** from the sidebar.
2. Upload a clear JPG or PNG with one main item.
3. Select **Analyse waste image** and review the predicted class, scores, and
   confidence band.
4. Optionally submit feedback, then visit the dashboard or prediction history.
5. Download the session history as CSV if needed.
6. Use **Smart Disposal Assistant** for general guidance and check local rules.

Prediction history and feedback are stored only in the current Streamlit
session; they are not a persistent database.

## Tests and development scripts

Run the current automated unit tests with:

```powershell
python -m unittest discover -s tests -v
```

`tests/test_model.py` is a separate model-architecture shape check; it builds a
fresh architecture and may download MobileNetV2 ImageNet weights. It does not
load the saved checkpoint. `tests/test_installation.py` is a manual dependency
checker.

The optional dataset preparation, training, and evaluation scripts are
`src/data_preprocessing.py`, `train.py`, and `evaluate.py`. They are not needed
to run the saved model in the Streamlit app.

## Limitations

- The classifier can be wrong, especially for cluttered, low-quality, or
  unfamiliar images. Softmax scores and the displayed confidence bands are
  not calibrated guarantees of correctness.
- The ImageNet label screen checks a limited set of top labels. It can miss
  non-waste images and can reject waste images; it is not reliable general
  non-waste detection.
- The dataset is limited and imbalanced, particularly in the trash category.
- Disposal rules vary by location and item condition. The app's guidance is
  general and does not replace local waste-authority instructions.
- The first use of the separate ImageNet screen may need internet access to
  download its pretrained weights if they are not already cached.

## Future improvements

Potential improvements include collecting more balanced and diverse labeled
images, testing non-waste screening with a representative held-out dataset,
calibrating confidence, and providing verified local disposal information.

