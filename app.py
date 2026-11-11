"""Polished Streamlit interface for the TrashSense 2.0 classifier.

Run from the project root with:
    venv/Scripts/python.exe -m streamlit run app.py
"""

from datetime import datetime
import os
import tempfile

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

CATEGORY_ICONS = {
    "cardboard": "📦",
    "glass": "🫙",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "♻️",
    "trash": "🗑️",
}

CATEGORY_DESCRIPTIONS = {
    "cardboard": "Boxes, cartons, and corrugated packaging",
    "glass": "Bottles and jars for local collection",
    "metal": "Cans and common metal containers",
    "paper": "Clean, dry sheets and paper products",
    "plastic": "Everyday plastic packaging and containers",
    "trash": "Items for the general-waste stream",
}

NAVIGATION_ICONS = {
    "Home": "⌂",
    "Waste Detection": "◉",
    "Smart Disposal Assistant": "♻",
    "Dashboard": "▦",
    "Prediction History": "◷",
    "About": "ⓘ",
}


@st.cache_resource
def get_prediction_resources():
    """Load the Stage 2 mapping and Stage 4 model once per Streamlit server."""
    class_names = load_class_names()
    model = load_trained_model(class_names)
    return model, class_names


def initialise_session_state():
    """Create empty session-only data the first time a browser opens the app."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "latest_prediction_id" not in st.session_state:
        st.session_state.latest_prediction_id = None
    if "page_navigation" not in st.session_state:
        st.session_state.page_navigation = "Home"


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


def apply_custom_style():
    """Add a small, responsive visual system while retaining native widgets."""
    st.markdown(
        """
        <style>
            :root {
                --forest: #164B32;
                --leaf: #218653;
                --mint: #E6F4EB;
                --mist: #F5F8F5;
                --line: #E2EBE4;
                --ink: #26352C;
                --muted: #718078;
            }

            .stApp, [data-testid="stAppViewContainer"] {
                background: #F5F8F5 !important;
                color: var(--ink);
            }
            .main .block-container {
                max-width: 1100px;
                padding-top: 2.3rem;
                padding-bottom: 3rem;
            }
            h1, h2, h3 {
                color: var(--forest) !important;
                font-family: "Avenir Next", "Segoe UI", sans-serif;
                letter-spacing: -0.025em;
            }
            h1 { font-size: clamp(1.9rem, 3vw, 2.6rem) !important; }
            h2 { margin-top: 1.65rem !important; }
            [data-testid="stSidebar"] > div:first-child {
                background: #FFFFFF;
                border-right: 1px solid var(--line);
            }
            [data-testid="stSidebar"] .stRadio label {
                border-radius: 9px;
                margin: 0.2rem 0;
                padding: 0.48rem 0.6rem;
                color: var(--ink) !important;
                transition: background-color 0.18s ease, transform 0.18s ease;
            }
            [data-testid="stSidebar"] .stRadio label p,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
                color: var(--muted) !important;
            }
            [data-testid="stSidebar"] .stRadio label:has(input:checked) {
                background: var(--mint);
                border-left: 3px solid var(--leaf);
                color: var(--forest) !important;
                font-weight: 700;
            }
            [data-testid="stSidebar"] input[type="radio"] { accent-color: var(--leaf); }
            [data-testid="stSidebar"] .stRadio label:hover {
                background: #EFF8F1;
                transform: translateX(2px);
            }
            .stButton > button, .stDownloadButton > button {
                border: 1px solid #218653;
                border-radius: 9px;
                background: #218653;
                color: #ffffff;
                font-weight: 650;
                padding: 0.5rem 1rem;
                box-shadow: 0 3px 10px rgba(33, 134, 83, 0.16);
                transition: transform 0.15s ease, background-color 0.15s ease;
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                background: #164B32;
                border-color: #164B32;
                color: #ffffff;
                transform: translateY(-1px);
            }
            .stButton > button:disabled {
                background: #E6F4EB;
                border-color: #E2EBE4;
                color: #718078;
                box-shadow: none;
            }
            [data-testid="stMetric"] {
                padding: 1rem;
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 14px;
                box-shadow: 0 5px 16px rgba(23, 65, 41, 0.06);
            }
            [data-testid="stMetricLabel"] { color: var(--muted); }
            [data-testid="stMetricValue"] { color: var(--forest); }
            [data-testid="stExpander"] {
                margin-bottom: 0.65rem;
                border: 1px solid var(--line) !important;
                border-radius: 12px !important;
                background: #FFFFFF;
                overflow: hidden;
            }
            [data-testid="stExpander"] summary { font-weight: 650; color: var(--forest); }
            [data-testid="stFileUploader"] {
                padding: 0.55rem;
                border: 1.5px dashed #8FC9A5;
                border-radius: 14px;
                background: #FFFFFF;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 12px;
                overflow: hidden;
            }
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stSelectbox"] > div > div,
            [data-testid="stFileUploader"] section {
                background: #FFFFFF !important;
                color: var(--ink) !important;
                border-color: var(--line) !important;
            }
            [data-testid="stProgress"] > div > div > div {
                background-color: var(--leaf);
            }
            .hero-panel {
                padding: clamp(1.5rem, 4vw, 3rem);
                margin: 0.2rem 0 2rem;
                border: 1px solid #D4E9DB;
                border-radius: 20px;
                background: linear-gradient(135deg, #FFFFFF 0%, #E6F4EB 100%);
                box-shadow: 0 12px 28px rgba(22, 75, 50, 0.07);
            }
            .eyebrow {
                margin: 0 0 0.55rem;
                color: var(--leaf);
                font-size: 0.78rem;
                font-weight: 750;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .hero-panel h1 {
                margin: 0 0 0.65rem;
                font-size: clamp(2rem, 5vw, 3.6rem) !important;
                line-height: 1.06;
            }
            .hero-copy {
                max-width: 690px;
                margin: 0;
                color: #4F6257;
                font-size: 1.08rem;
                line-height: 1.65;
            }
            .section-kicker {
                margin-bottom: -0.65rem;
                color: var(--leaf);
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.07em;
                text-transform: uppercase;
            }
            .category-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin: 1rem 0 1.3rem;
            }
            .category-card {
                min-height: 96px;
                padding: 1rem;
                border: 1px solid var(--line);
                border-radius: 14px;
                background: #FFFFFF;
                box-shadow: 0 4px 12px rgba(22, 75, 50, 0.05);
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            }
            .category-card:hover {
                transform: translateY(-3px);
                border-color: #B9DDC5;
                box-shadow: 0 9px 20px rgba(22, 75, 50, 0.10);
            }
            .category-icon { font-size: 1.35rem; }
            .category-name {
                display: block;
                margin-top: 0.35rem;
                color: var(--forest);
                font-size: 0.98rem;
                font-weight: 700;
            }
            .category-caption { color: var(--muted); font-size: 0.8rem; }
            .info-card, .result-card {
                padding: 1.25rem;
                margin: 1rem 0;
                border: 1px solid var(--line);
                border-radius: 15px;
                background: #FFFFFF;
                box-shadow: 0 5px 16px rgba(22, 75, 50, 0.05);
            }
            .result-label {
                margin: 0.15rem 0 0;
                color: var(--forest);
                font-size: clamp(1.5rem, 4vw, 2.2rem);
                font-weight: 750;
                line-height: 1.15;
            }
            .result-overline {
                margin: 0;
                color: var(--leaf);
                font-size: 0.77rem;
                font-weight: 750;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .guidance-label {
                display: block;
                margin-bottom: 0.12rem;
                color: var(--forest);
                font-size: 0.82rem;
                font-weight: 700;
            }
            .upload-callout {
                padding: 1.2rem;
                margin: 0.8rem 0;
                border: 1px dashed #8FC9A5;
                border-radius: 14px;
                background: #FFFFFF;
                color: var(--muted);
                text-align: center;
            }
            .upload-callout strong { color: var(--forest); }
            .about-card {
                height: 100%;
                padding: 1.2rem;
                border: 1px solid var(--line);
                border-radius: 14px;
                background: #FFFFFF;
                box-shadow: 0 5px 16px rgba(22, 75, 50, 0.05);
            }
            .about-card h3 {
                margin: 0 0 0.5rem;
                color: var(--forest) !important;
                font-size: 1rem;
            }
            .about-card p { margin: 0; color: var(--muted); line-height: 1.6; }
            @media (max-width: 640px) {
                .main .block-container { padding: 1.2rem 1rem 2.5rem; }
                .category-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .hero-panel { border-radius: 16px; }
                [data-testid="stMetric"] { margin-bottom: 0.45rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_confidence_indicator(prediction_record):
    """Display an honest confidence band without claiming certainty."""
    confidence = prediction_record["confidence"]
    level = prediction_record["confidence_level"]
    confidence_column, level_column = st.columns(2)
    confidence_column.metric("Model confidence", f"{confidence:.2%}")
    level_column.metric("Confidence level", level)

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
    st.markdown('<div class="section-kicker">Classification result</div>', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="result-card">
            <p class="result-overline">Predicted waste category</p>
            <p class="result-label">{CATEGORY_ICONS[prediction_record["predicted_label"]]} {prediction_record["predicted_label"].capitalize()}</p>
        </div>''',
        unsafe_allow_html=True,
    )
    show_confidence_indicator(prediction_record)

    st.markdown("#### Category probabilities")
    st.caption("Actual Softmax scores from the trained model, ordered from highest to lowest.")
    sorted_scores = sorted(
        prediction_record["scores"].items(), key=lambda score: score[1], reverse=True
    )
    for class_name, probability in sorted_scores:
        label_column, score_column = st.columns([3, 1])
        label_column.write(f"**{class_name.capitalize()}**")
        score_column.write(f"{probability:.2%}")
        st.progress(int(round(probability * 100)))

    st.markdown("#### Help us improve")
    if prediction_record["feedback"] == "Not provided":
        st.caption("Was this prediction correct? Your response is saved only in this session.")
        correct_column, incorrect_column = st.columns(2)
        if correct_column.button(
            "Prediction correct", key=f"correct_{prediction_record['id']}", use_container_width=True
        ):
            update_feedback(prediction_record["id"], "Correct")
            st.success("Feedback saved. It does not automatically retrain the model.")
        if incorrect_column.button(
            "Prediction incorrect", key=f"incorrect_{prediction_record['id']}", use_container_width=True
        ):
            update_feedback(prediction_record["id"], "Incorrect")
            st.info("Feedback saved. It can be reviewed later to improve a future dataset.")
    else:
        st.caption(f"Your feedback: {prediction_record['feedback']}")


def navigate_to_detection():
    """Open the detection workspace from the home-page call to action."""
    st.session_state.page_navigation = "Waste Detection"


def show_home_page():
    """Show a concise introduction and the six supported classes."""
    st.markdown(
        """
        <section class="hero-panel">
            <p class="eyebrow">Clearer sorting, one photo at a time</p>
            <h1>Meet TrashSense 2.0</h1>
            <p class="hero-copy">A simple AI-assisted tool for identifying everyday waste from an image and finding a sensible next step for disposal.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">What it recognises</div>', unsafe_allow_html=True)
    st.markdown("## Six everyday waste categories")
    category_cards = "".join(
        f'''<div class="category-card">
            <span class="category-icon">{CATEGORY_ICONS[class_name]}</span>
            <span class="category-name">{class_name.capitalize()}</span>
            <span class="category-caption">{CATEGORY_DESCRIPTIONS[class_name]}</span>
        </div>'''
        for class_name in config.WASTE_CATEGORIES
    )
    st.markdown(f'<div class="category-grid">{category_cards}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card"><strong>Ready when you are.</strong><br>'
        'Choose a clear photo with one main item, then let the trained model estimate its category.</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Start waste detection",
        type="primary",
        use_container_width=True,
        on_click=navigate_to_detection,
    )
    st.caption(
        "TrashSense provides guidance, not a guarantee. Always follow your local waste-management rules."
    )


def show_detection_page():
    """Accept an upload, call predict.py, and record the real model result."""
    st.markdown('<div class="section-kicker">Image classifier</div>', unsafe_allow_html=True)
    st.header("Waste Detection")
    st.write("Upload a clear image containing one waste item to receive a model-based category estimate.")
    st.caption(
        "Confidence bands: High at or above "
        f"{config.HIGH_CONFIDENCE_THRESHOLD:.0%}; Medium from "
        f"{config.MEDIUM_CONFIDENCE_THRESHOLD:.0%} to below "
        f"{config.HIGH_CONFIDENCE_THRESHOLD:.0%}; Low below "
        f"{config.MEDIUM_CONFIDENCE_THRESHOLD:.0%}."
    )
    st.markdown(
        '<div class="upload-callout"><strong>⇧ Upload an image</strong><br>'
        'Drag and drop a clear JPG or PNG photo, or browse to choose one. One well-lit item works best.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="For the best result, use a well-lit image with one main item.",
    )

    if uploaded_file is not None:
        try:
            with Image.open(uploaded_file) as preview_image:
                preview_copy = preview_image.copy()
                image_format = preview_image.format or "Unknown"
                image_width, image_height = preview_image.size

            preview_column, details_column = st.columns([3, 2], gap="large")
            with preview_column:
                st.image(preview_copy, caption="Image preview", use_column_width=True)
            with details_column:
                st.markdown("#### Image details")
                st.metric("Dimensions", f"{image_width} × {image_height} px")
                st.metric("Format", image_format.upper())
                st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
                st.caption("For a more reliable estimate, avoid cluttered or very dark photos.")
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be opened as a valid image.")

    if st.button(
        "Analyse waste image",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Analysing the image with the trained model..."):
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
        st.divider()
        show_prediction_result(latest_prediction)


def show_guidance_page():
    """Show cautious, structured disposal guidance for the six model categories."""
    st.markdown('<div class="section-kicker">Practical next steps</div>', unsafe_allow_html=True)
    st.header("Smart Disposal Assistant")
    st.write("Use the category below as a starting point, then check the requirements of your local collection service.")
    st.warning(
        "Rules differ by city, country, facility, and item condition. A category prediction does not prove that an item is recyclable. Follow your local waste-management authority's instructions."
    )

    for class_name in config.WASTE_CATEGORIES:
        guidance = DISPOSAL_GUIDANCE[class_name]
        with st.expander(f"{CATEGORY_ICONS[class_name]}  {class_name.capitalize()} disposal guidance"):
            st.markdown(
                f'''<span class="guidance-label">General suggestion</span>{guidance["general"]}
                <br><br><span class="guidance-label">Before you dispose of it</span>{guidance["prepare"]}
                <br><br><span class="guidance-label">Common mistake to avoid</span>{guidance["common_mistakes"]}
                <br><br><span class="guidance-label">Check local rules</span>{guidance["local_rules"]}''',
                unsafe_allow_html=True,
            )


def show_dashboard_page():
    """Calculate analytics only from real predictions in session history."""
    st.markdown('<div class="section-kicker">Your current session</div>', unsafe_allow_html=True)
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
    average_confidence = history_df["Confidence (%)"].mean()

    total_column, frequent_column, confidence_column = st.columns(3)
    total_column.metric("Total predictions", len(history_df))
    frequent_column.metric("Most detected", f"{most_frequent_label} ({most_frequent_count})")
    confidence_column.metric("Average confidence", f"{average_confidence:.1f}%")

    st.markdown("#### Predictions by category")
    st.caption("Counts are calculated from real predictions made in this browser session.")
    st.bar_chart(
        category_counts.to_frame(name="Predictions"),
        color="#2f8657",
        use_container_width=True,
    )

    st.markdown("#### Recent predictions")
    st.dataframe(
        history_df.iloc[::-1].head(10),
        use_container_width=True,
        hide_index=True,
    )


def show_history_page():
    """Display, download, and clear the current session's real prediction history."""
    st.markdown('<div class="section-kicker">Session records</div>', unsafe_allow_html=True)
    st.header("Prediction History")
    history_df = history_to_dataframe()

    if history_df.empty:
        st.info("Your prediction history is empty. Make a prediction to create a record.")
        return

    st.caption("This history is stored only for the current browser session.")
    st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    download_column, clear_column = st.columns(2)
    with download_column:
        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="trashsense_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with clear_column:
        if st.button("Clear current session history", use_container_width=True):
            st.session_state.prediction_history = []
            st.session_state.latest_prediction_id = None
            st.success("Current session history cleared.")


def show_about_page():
    """Explain the project design, scope, and limitations."""
    st.markdown('<div class="section-kicker">Project overview</div>', unsafe_allow_html=True)
    st.header("About TrashSense 2.0")

    about_sections = [
        (
            "Problem statement",
            "Waste can be difficult to sort correctly from appearance alone. TrashSense explores image classification as decision support before an item enters a waste stream.",
        ),
        (
            "Project objective",
            "Classify an uploaded image into cardboard, glass, metal, paper, plastic, or trash, then offer cautious general disposal guidance.",
        ),
        (
            "Model architecture",
            "ImageNet-pretrained MobileNetV2 is used as a CNN feature extractor, followed by global average pooling, a 128-unit ReLU layer, 30% dropout, and a six-class Softmax output.",
        ),
        (
            "Technology stack",
            "Python, TensorFlow/Keras, Streamlit, Pillow, Pandas, and NumPy support the trained model, interface, image handling, and session analytics.",
        ),
        (
            "Prototype limitations",
            "Results depend on image quality and the training data. The model knows six categories only and cannot replace local recycling requirements or facility acceptance rules.",
        ),
        (
            "Future improvements",
            "Future versions could use more diverse data, review feedback with verified labels, calibrate confidence, and connect users to trusted local disposal guidance.",
        ),
    ]
    for first, second in zip(about_sections[::2], about_sections[1::2]):
        left_column, right_column = st.columns(2)
        for column, (title, description) in ((left_column, first), (right_column, second)):
            with column:
                st.markdown(
                    f'<div class="about-card"><h3>{title}</h3><p>{description}</p></div>',
                    unsafe_allow_html=True,
                )


def main():
    """Configure the app, initialise session data, and show the selected page."""
    st.set_page_config(
        page_title="TrashSense 2.0",
        page_icon="♻️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialise_session_state()
    apply_custom_style()

    st.sidebar.markdown("## ♻️ TrashSense")
    st.sidebar.caption("AI-assisted waste identification")
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
        key="page_navigation",
        format_func=lambda page_name: f"{NAVIGATION_ICONS[page_name]}  {page_name}",
    )
    st.sidebar.divider()
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
