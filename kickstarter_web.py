

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from datetime import timedelta

st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="wide"
)

BASE = Path(__file__).parent
MODELS = BASE / "models"

@st.cache_resource
def load_models():
    return {
        "preprocessor": joblib.load(MODELS / "metadata_preprocessor.pkl"),
        "tfidf": joblib.load(MODELS / "tfidf_vectorizer.pkl"),
        "nb": joblib.load(MODELS / "naive_bayes.pkl"),
        "count": joblib.load(MODELS / "count_vectorizer.pkl"),
        "lda": joblib.load(MODELS / "lda_model.pkl"),
        "gb": joblib.load(MODELS / "gradient_boosting.pkl"),
    }

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🚀 Kickstarter Success Predictor")
st.write("Estimate campaign success using your trained machine-learning model.")
st.warning(
    "This model was trained on synthetic project data. "
    "Its predictions are estimates, not guarantees of real-world success."
)

with st.sidebar:
    st.header("Campaign Details")
    name = st.text_input("Campaign name", "Smart Eco Travel Bottle")
    blurb = st.text_area(
        "Campaign description",
        "A reusable smart bottle designed to help travelers reduce plastic waste."
    )
    goal = st.number_input("Funding goal (USD)", min_value=1.0, value=5000.0)
    duration = st.slider("Duration (days)", 1, 90, 30)
    launch_date = st.date_input(
        "Launch date",
        value=pd.Timestamp.today().date()
    )
    currency = st.selectbox("Currency", ["USD", "GBP", "EUR", "CAD", "AUD"])
    country = st.selectbox(
        "Country code",
        ["US", "GB", "CA", "AU", "DE", "FR", "NL", "DK", "IN", "Other"]
    )
    parent = st.selectbox(
        "Main category",
        ["Technology", "Art", "Film & Video", "Games", "Music",
         "Publishing", "Design", "Food", "Photography", "Theater",
         "Dance", "Journalism", "Comics", "Other"]
    )
    category = st.text_input("Specific category", "Gadgets")
    location = st.text_input("City or location", "New York")
    location_type = st.selectbox(
        "Location type", ["City", "Town", "County", "Other"]
    )

if st.button("Predict Campaign Success", type="primary"):
    try:
        models = load_models()
        launch = pd.to_datetime(launch_date)
        deadline = launch + timedelta(days=duration)

        row = pd.DataFrame([{
            "name_length": len(name),
            "blurb_length": len(blurb),
            "goal_usd": float(goal),
            "duration_days": int(duration),
            "launch_year": launch.year,
            "launch_month": launch.month,
            "launch_dayofweek": launch.dayofweek,
            "currency": currency,
            "country": country,
            "parent_category": parent,
            "category": category,
            "location": location,
            "location_type": location_type,
        }])

        metadata = models["preprocessor"].transform(row)
        text = clean_text(blurb)

        tfidf_features = models["tfidf"].transform([text])
        nb_probability = models["nb"].predict_proba(
            tfidf_features
        )[0, 1]

        count_features = models["count"].transform([text])
        topics = models["lda"].transform(count_features)

        features = np.column_stack([
            metadata,
            [nb_probability],
            topics
        ])

        probability = models["gb"].predict_proba(features)[0, 1]

        st.subheader("Prediction Result")
        if probability >= 0.5:
            st.success("Predicted outcome: SUCCESS 🎉")
        else:
            st.error("Predicted outcome: FAILURE")

        col1, col2 = st.columns(2)
        col1.metric("Success probability", f"{probability:.1%}")
        col2.metric("Failure probability", f"{1-probability:.1%}")
        st.progress(float(probability))
        st.caption(
            f"Launch: {launch.date()} | Deadline: {deadline.date()}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check that all six .pkl files are in the same folder "
            "as kickstarter_web.py and that the installed libraries "
            "are compatible with the saved models."
        )

with st.expander("Explore project dataset"):
    csv_path = BASE / "kickstarter_success_prediction_dataset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.write(f"{len(df):,} rows | {len(df.columns)} columns")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Dataset CSV not found.")
