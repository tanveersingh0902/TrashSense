"""Streamlit web interface for TrashSense 2.0.

Run from the project root with:
    venv\Scripts\python.exe -m streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from predict import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_class_names,
    load_trained_model,
    predict_image,
)


DISPOSAL_GUIDANCE = {
    "cardboard": (
        "Keep cardboard clean and dry, flatten it, and check whether your local "
        "collection accepts it. Food-soiled, wax-coated, or heavily contaminated "
        "cardboard may need a different disposal route."
    ),
    "glass": (
        "Empty containers and follow local instructions about rinsing, lids, and "
        "colour sorting. Handle broken glass carefully and follow local guidance; "
        "do not put it loosely where it could injure someone."
    ),
    "metal": (
        "Empty metal containers and check local rules before recycling them. "
        "Large metal objects, sharp items, and mixed-material products may require "
        "a special collection service."
    ),
    "paper": (
        "Keep paper clean and dry. Local facilities may reject paper that is wet, "
        "food-soiled, plastic-coated, or mixed with other materials."
    ),
    "plastic": (
        "Check the item and your local rules: plastic type, shape, and condition "
        "affect whether it is accepted. Empty and clean items only when local "
        "guidance requests it; many plastics are not recyclable everywhere."
    ),
    "trash": (
        "Use the local general-waste route for items that cannot be reused or "
        "accepted for recycling. Batteries, electronics, chemicals, medical waste, "
        "and other hazardous items need official specialist collection, not normal trash."
    ),
}


@st.cache_resource
def get_prediction_resources():
    """Load the class mapping and trained model once per Streamlit server."""
    class_names = load_class_names()
    model = load_trained_model(class_names)
    return model, class_names


def initialise_session_state():
    """Create empty, per-browser-session dashboard counters when needed."""
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
    if "prediction_counts" not in st.session_state:
        st.session_state.prediction_counts = {
            class_name: 0 for class_name in config.WASTE_CATEGORIES
        }


def predict_uploaded_file(model, uploaded_file, class_names):
    """Temporarily save an upload so the existing predict_image function can use it."""
    _, file_extension = os.path.splitext(uploaded_file.name)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=file_extension or ".png", delete=False) as file:
            file.write(uploaded_file.getvalue())
            temporary_path = file.name

        return predict_image(model, temporary_path, class_names)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def show_home_page():
    """Show the project introduction."""
    st.header("Welcome to TrashSense 2.0")
    st.write(
        "TrashSense 2.0 is an image-classification project that helps identify "
        "a waste category from a photo. It uses a convolutional neural network "
        "(CNN) based on MobileNetV2."
    )
    st.subheader("Supported categories")
    st.write(", ".join(class_name.capitalize() for class_name in config.WASTE_CATEGORIES))
    st.info(
        "A prediction is guidance, not a guarantee. Use the Waste Detection page "
        "with a clear, single-item image."
    )


def show_detection_page():
    """Accept an upload, call predict.py, and display the real model result."""
    st.header("Waste Detection")
    st.write("Upload a clear image containing one waste item.")

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        format="%.0f%%",
        help="Predictions below this score are labelled uncertain.",
    )
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], help="JPG and PNG files are supported."
    )

    if uploaded_file is not None:
        try:
            with Image.open(uploaded_file) as preview_image:
                st.image(preview_image.copy(), caption="Uploaded image", use_column_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be opened as a valid image.")

    if st.button("Predict waste category", disabled=uploaded_file is None):
        try:
            with st.spinner("Loading the trained model and analysing the image..."):
                model, class_names = get_prediction_resources()
                probabilities, predicted_label, confidence = predict_uploaded_file(
                    model, uploaded_file, class_names
                )

            st.subheader("Prediction result")
            st.write(f"**Predicted category:** {predicted_label.capitalize()}")
            st.write(f"**Confidence:** {confidence:.2%}")

            if confidence >= threshold:
                st.success("The confidence meets the selected threshold.")
            else:
                st.warning(
                    "Uncertain prediction: the highest score is below the selected "
                    f"{threshold:.0%} threshold. Try a clearer image or use manual review."
                )

            st.caption("Scores for all supported classes")
            for class_name, probability in zip(class_names, probabilities):
                st.write(f"- {class_name.capitalize()}: {probability:.2%}")

            st.session_state.total_predictions += 1
            st.session_state.prediction_counts[predicted_label] += 1
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))


def show_guidance_page():
    """Show cautious, general disposal guidance for the six model categories."""
    st.header("Disposal Guidance")
    st.warning(
        "Rules differ by city, country, facility, and item condition. A category "
        "prediction does not prove that an item is recyclable. Follow your local "
        "waste-management authority's instructions."
    )

    for class_name in config.WASTE_CATEGORIES:
        with st.expander(class_name.capitalize()):
            st.write(DISPOSAL_GUIDANCE[class_name])


def show_dashboard_page():
    """Show only real predictions made in this browser session."""
    st.header("Current Session Dashboard")
    st.metric("Images classified", st.session_state.total_predictions)

    if st.session_state.total_predictions == 0:
        st.info("No images have been classified in this session yet.")
        return

    chart_data = pd.DataFrame.from_dict(
        st.session_state.prediction_counts,
        orient="index",
        columns=["Predictions"],
    )
    chart_data.index.name = "Waste category"
    st.bar_chart(chart_data)
    st.dataframe(chart_data, use_container_width=True)


def show_about_page():
    """Explain the project design, scope, and limitations."""
    st.header("About TrashSense 2.0")

    st.subheader("Problem statement")
    st.write(
        "Waste can be difficult to sort correctly from appearance alone. TrashSense "
        "is a learning project that explores image classification as decision support."
    )

    st.subheader("Objective")
    st.write(
        "Classify an uploaded image into cardboard, glass, metal, paper, plastic, "
        "or trash, then provide cautious general disposal guidance."
    )

    st.subheader("Neural-network architecture")
    st.write(
        "The model uses ImageNet-pretrained MobileNetV2 as a CNN feature extractor, "
        "followed by global average pooling, a 128-unit ReLU layer, 30% dropout, "
        "and a six-class Softmax output layer."
    )

    st.subheader("Dataset and technology")
    st.write(
        "The TrashNet dataset was split into separate training, validation, and test "
        "sets. Python, TensorFlow/Keras, Pillow, NumPy, and Streamlit are used."
    )

    st.subheader("Limitations and future improvements")
    st.write(
        "Predictions depend on image quality and the training data. The model knows "
        "only six categories and does not replace local recycling rules. Future work "
        "could add more diverse data, improve difficult classes, calibrate confidence, "
        "and integrate verified local disposal information."
    )


def main():
    """Configure the app, initialise session data, and show the selected page."""
    st.set_page_config(page_title="TrashSense 2.0", page_icon="♻️", layout="centered")
    initialise_session_state()

    st.title("♻️ TrashSense 2.0")
    st.caption("AI-assisted waste image classification with MobileNetV2")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Waste Detection", "Disposal Guidance", "Dashboard", "About"],
    )
    st.sidebar.caption("Dashboard values are for the current browser session only.")

    if page == "Home":
        show_home_page()
    elif page == "Waste Detection":
        show_detection_page()
    elif page == "Disposal Guidance":
        show_guidance_page()
    elif page == "Dashboard":
        show_dashboard_page()
    else:
        show_about_page()


if __name__ == "__main__":
    main()
