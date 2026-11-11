"""Streamlit web interface for TrashSense 2.0.

Run from the project root with:
    venv\Scripts\python.exe -m streamlit run app.py
"""

from datetime import datetime
import os
import tempfile

# The deployed checkpoint was saved with Keras 2.15; load it through legacy Keras.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from predict import load_class_names, load_trained_model, predict_image
from utils.disposal_guidance import DISPOSAL_GUIDANCE


# These are the columns shown in the history table and downloaded CSV file.
HISTORY_COLUMNS = [
    "Date and time",
    "Predicted category",
    "Confidence (%)",
    "Confidence level",
    "User feedback",
]


@st.cache_resource
def get_prediction_resources():
    """Load the Stage 2 mapping and Stage 4 model once per Streamlit server."""
    class_names = load_class_names()
    model = load_trained_model(class_names)
    return model, class_names


def initialise_session_state():
    """Create empty session-only history data the first time a browser opens the app."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "latest_prediction_id" not in st.session_state:
        st.session_state.latest_prediction_id = None


def get_confidence_level(confidence):
    """Convert the highest Softmax score into a high, medium, or low label."""
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


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
        # Remove the temporary upload whether prediction succeeds or raises an error.
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def add_prediction_to_history(predicted_label, confidence, probabilities, class_names):
    """Store one real model result in the current browser session."""
    timestamp = datetime.now().astimezone()
    prediction_id = f"{timestamp.isoformat()}_{len(st.session_state.prediction_history)}"
    prediction_record = {
        "id": prediction_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "confidence_level": get_confidence_level(confidence),
        "feedback": "Not provided",
        "scores": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities)
        },
    }
    st.session_state.prediction_history.append(prediction_record)
    st.session_state.latest_prediction_id = prediction_id
    return prediction_record


def get_latest_prediction():
    """Find the most recent prediction record, or return None when there is none."""
    latest_id = st.session_state.latest_prediction_id
    for prediction_record in reversed(st.session_state.prediction_history):
        if prediction_record["id"] == latest_id:
            return prediction_record
    return None


def update_feedback(prediction_id, feedback):
    """Save a user's correct or incorrect response in its matching history record."""
    for prediction_record in st.session_state.prediction_history:
        if prediction_record["id"] == prediction_id:
            prediction_record["feedback"] = feedback
            return


def history_to_dataframe():
    """Turn session-history dictionaries into a displayable and downloadable table."""
    rows = [
        {
            "Date and time": prediction_record["timestamp"],
            "Predicted category": prediction_record["predicted_label"].capitalize(),
            "Confidence (%)": round(prediction_record["confidence"] * 100, 2),
            "Confidence level": prediction_record["confidence_level"],
            "User feedback": prediction_record["feedback"],
        }
        for prediction_record in st.session_state.prediction_history
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def show_confidence_indicator(prediction_record):
    """Display an honest confidence band without claiming certainty."""
    confidence = prediction_record["confidence"]
    level = prediction_record["confidence_level"]
    st.write(f"**Confidence:** {confidence:.2%} ({level})")

    if level == "High":
        st.success(
            "High confidence according to the model's Softmax score. This is not a guarantee of correctness."
        )
    elif level == "Medium":
        st.info(
            "Medium confidence. Check that the image is clear and contains one item before relying on the result."
        )
    else:
        st.warning(
            "Low confidence. Please verify this result, try a clearer image, or use manual review."
        )


def show_prediction_result(prediction_record):
    """Display one saved model result and collect optional user feedback."""
    st.subheader("Latest prediction")
    st.write(
        f"**Predicted category:** {prediction_record['predicted_label'].capitalize()}"
    )
    show_confidence_indicator(prediction_record)

    st.caption("Actual Softmax scores from the model")
    for class_name, probability in prediction_record["scores"].items():
        st.write(f"- {class_name.capitalize()}: {probability:.2%}")

    if prediction_record["feedback"] == "Not provided":
        st.write("Was this prediction correct?")
        correct_column, incorrect_column = st.columns(2)
        if correct_column.button("Prediction correct", key=f"correct_{prediction_record['id']}"):
            update_feedback(prediction_record["id"], "Correct")
            st.success("Feedback saved. It does not automatically retrain the model.")
        if incorrect_column.button("Prediction incorrect", key=f"incorrect_{prediction_record['id']}"):
            update_feedback(prediction_record["id"], "Incorrect")
            st.info("Feedback saved. It can be reviewed later to improve a future dataset.")
    else:
        st.caption(f"Your feedback: {prediction_record['feedback']}")


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
    """Accept an upload, call predict.py, and record the real model result."""
    st.header("Waste Detection")
    st.write("Upload a clear image containing one waste item.")
    st.caption(
        "Confidence bands: High at or above "
        f"{config.HIGH_CONFIDENCE_THRESHOLD:.0%}; Medium from "
        f"{config.MEDIUM_CONFIDENCE_THRESHOLD:.0%} to below "
        f"{config.HIGH_CONFIDENCE_THRESHOLD:.0%}; Low below "
        f"{config.MEDIUM_CONFIDENCE_THRESHOLD:.0%}."
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
            add_prediction_to_history(
                predicted_label, confidence, probabilities, class_names
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))

    latest_prediction = get_latest_prediction()
    if latest_prediction is not None:
        show_prediction_result(latest_prediction)


def show_guidance_page():
    """Show cautious, structured disposal guidance for the six model categories."""
    st.header("Smart Disposal Assistant")
    st.warning(
        "Rules differ by city, country, facility, and item condition. A category "
        "prediction does not prove that an item is recyclable. Follow your local "
        "waste-management authority's instructions."
    )

    for class_name in config.WASTE_CATEGORIES:
        guidance = DISPOSAL_GUIDANCE[class_name]
        with st.expander(class_name.capitalize()):
            st.write(f"**General suggestion:** {guidance['general']}")
            st.write(f"**Preparation:** {guidance['prepare']}")
            st.write(f"**Common mistake:** {guidance['common_mistakes']}")
            st.write(f"**Check local rules:** {guidance['local_rules']}")


def show_dashboard_page():
    """Calculate analytics only from real predictions in session history."""
    st.header("Waste Analytics Dashboard")
    history_df = history_to_dataframe()

    if history_df.empty:
        st.info("No predictions yet. Classify an image to see real session analytics.")
        return

    category_order = [class_name.capitalize() for class_name in config.WASTE_CATEGORIES]
    category_counts = (
        history_df["Predicted category"].value_counts().reindex(category_order, fill_value=0)
    )
    most_frequent_count = int(category_counts.max())
    most_frequent_categories = category_counts[
        category_counts == most_frequent_count
    ].index.tolist()
    most_frequent_label = ", ".join(most_frequent_categories)

    total_column, frequent_column = st.columns(2)
    total_column.metric("Total predictions", len(history_df))
    frequent_column.metric(
        "Most frequently detected", f"{most_frequent_label} ({most_frequent_count})"
    )

    st.subheader("Predictions by category")
    st.bar_chart(category_counts.to_frame(name="Predictions"))

    st.subheader("Recent prediction history")
    st.dataframe(history_df.iloc[::-1].head(10), use_container_width=True)


def show_history_page():
    """Display, download, and clear the current session's real prediction history."""
    st.header("Prediction History")
    history_df = history_to_dataframe()

    if history_df.empty:
        st.info("Your prediction history is empty. Make a prediction to create a record.")
        return

    st.dataframe(history_df.iloc[::-1], use_container_width=True)
    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv_data,
        file_name="trashsense_prediction_history.csv",
        mime="text/csv",
    )

    if st.button("Clear current session history"):
        st.session_state.prediction_history = []
        st.session_state.latest_prediction_id = None
        st.success("Current session history cleared.")


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
        "sets. Python, TensorFlow/Keras, Pillow, NumPy, Pandas, and Streamlit are used."
    )

    st.subheader("Limitations and future improvements")
    st.write(
        "Predictions depend on image quality and the training data. The model knows "
        "only six categories and does not replace local recycling rules. Future work "
        "could add more diverse data, review feedback with real labels, calibrate "
        "confidence, and integrate verified local disposal information."
    )


def main():
    """Configure the app, initialise session data, and show the selected page."""
    st.set_page_config(page_title="TrashSense 2.0", layout="centered")
    initialise_session_state()

    st.title("TrashSense 2.0")
    st.caption("AI-assisted waste image classification with MobileNetV2")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Waste Detection",
            "Smart Disposal Assistant",
            "Dashboard",
            "Prediction History",
            "About",
        ],
    )
    st.sidebar.caption("History and analytics are kept only for this browser session.")

    if page == "Home":
        show_home_page()
    elif page == "Waste Detection":
        show_detection_page()
    elif page == "Smart Disposal Assistant":
        show_guidance_page()
    elif page == "Dashboard":
        show_dashboard_page()
    elif page == "Prediction History":
        show_history_page()
    else:
        show_about_page()


if __name__ == "__main__":
    main()
