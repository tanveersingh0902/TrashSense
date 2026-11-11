# 🗑️ TRASHSENSE 2.0

**AI-Powered Waste Classification System using CNN and Transfer Learning**

## 📖 About

TrashSense 2.0 is a deep learning project that classifies waste images into 6 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

It uses **MobileNetV2** (a pre-trained CNN) with **Transfer Learning** to achieve high accuracy even with a limited dataset.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow/Keras | Deep learning framework |
| MobileNetV2 | Pre-trained CNN model |
| Streamlit | Web application interface |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Plotting graphs |
| Scikit-learn | ML utilities |
| Pillow | Image processing |

## 📁 Project Structure

```
TrashSense/
├── data/                   # Dataset storage
│   ├── raw/                # Original unprocessed images
│   └── processed/          # Preprocessed images
├── models/                 # Model-related code
│   ├── model.py            # MobileNetV2 waste-classification model
│   └── saved_models/       # Trained model files (.keras)
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Main source code
├── utils/                  # Helper/utility functions
├── app/                    # Streamlit web application
├── tests/                  # Test files
├── outputs/                # Generated outputs
│   ├── plots/              # Training charts and graphs
│   └── reports/            # Classification reports
├── config.py               # Project settings and constants
├── requirements.txt        # Python library dependencies
└── README.md               # This file
```

## 🧠 Stage 3: Build and verify the model

The model uses ImageNet-pretrained MobileNetV2 features with a new six-class
TrashSense classification head. Its pretrained layers are frozen initially.
The model expects 224 × 224 RGB images preprocessed with MobileNetV2's
`preprocess_input`, as configured in `src/data_preprocessing.py`.

To print the model summary and check its input/output shapes without training:

```bash
venv\Scripts\python.exe -m tests.test_model
```

The first run may download MobileNetV2's pretrained ImageNet weights if they
are not already cached.

## 🚀 Setup Instructions

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python tests/test_installation.py
```

## 👤 Author

**Tanveer Singh** — B.Tech Computer Science, 2nd Year
