import streamlit as st
import sys
from pathlib import Path
import joblib

# allow imports from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from save_model import load_model
from predict import make_prediction


# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction")
st.write(
    "Predict whether a flight will be delayed by **15 minutes or more** "
    "based on historical flight data."
)


# Load model and valid categories
@st.cache_resource
def load_resources():
    model = load_model("models/flight_delay_model.pkl")
    categories = joblib.load("models/valid_categories.pkl")
    return model, categories


model, valid_categories = load_resources()

# Input form
st.subheader("Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline Code",
        valid_categories["airlines"]
    )

    origin = st.selectbox(
        "Origin Airport",
        valid_categories["origins"]
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=50,
        max_value=5000,
        value=500
    )

with col2:
    destination = st.selectbox(
        "Destination Airport",
        valid_categories["destinations"]
    )

    dep_hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=10
    )

    day_of_week = st.selectbox(
        "Day of Week",
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x - 1]
    )

month = st.selectbox("Month", list(range(1, 13)))

# Prediction
if st.button("Predict Delay"):
    input_data = {
        "AIRLINE": airline,
        "ORIGIN": origin,
        "DEST": destination,
        "DISTANCE": distance,
        "dep_hour": dep_hour,
        "DAY_OF_WEEK": day_of_week,
        "MONTH": month
    }

    prediction, probability = make_prediction(
        model,
        input_data,
        threshold=0.35
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Flight is likely to be delayed")
    else:
        st.success("🟢 Flight is likely to be on time")

    st.metric("Delay Probability", f"{probability:.1%}")
    
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px;">
        Built by <b>Qaleel Sha Backer</b> |
        <a href="https://www.linkedin.com/in/qaleel-sha-backer/" target="_blank">
            LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True
)