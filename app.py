"""Sri Lankan tourist arrival trend classifier: Streamlit interface."""

from pathlib import Path
import base64
import pickle

import pandas as pd
import streamlit as st


MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MODEL_FILE = Path(__file__).with_name("tourist_trend_model.pkl")
BACKGROUND_FILE = Path(__file__).with_name("sri_lanka_background.jpg")

st.set_page_config(
    page_title="Sri Lanka Tourist Arrival Trends",
    page_icon="🇱🇰",
    layout="wide",
)



def style_page():
    """Apply travel-inspired colors and the supplied Sri Lankan photo."""
    background = "linear-gradient(120deg, #073a49, #175b68 58%, #26735b)"
    if BACKGROUND_FILE.exists():
        encoded = base64.b64encode(BACKGROUND_FILE.read_bytes()).decode("ascii")
        background = (
            "linear-gradient(135deg, rgba(5, 22, 38, 0.56), "
            "rgba(5, 36, 46, 0.56)), "
            f"url('data:image/jpeg;base64,{encoded}')"
        )

    st.markdown(
        "<style>" + """
        /* Tourism photo and subdued overlay across the entire app */
        .stApp {
            background-image: BACKGROUND_PLACEHOLDER;
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stHeader"],
        [data-testid="stAppViewContainer"],
        [data-testid="stBottomBlockContainer"] {
            background: transparent;
        }
        /* Translucent content surface keeps the photo visible but text legible */
        [data-testid="stMainBlockContainer"] {
            max-width: 960px;
            margin: 30px auto 54px;
            padding: 2.6rem 3.0rem 3.3rem;
            background: rgba(250, 253, 255, 0.94);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 26px;
            box-shadow: 0 20px 70px rgba(2, 22, 38, 0.32);
            backdrop-filter: blur(9px);
            -webkit-backdrop-filter: blur(9px);
        }
        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3 {
            color: #083d52;
            letter-spacing: -0.02em;
        }
        [data-testid="stMainBlockContainer"] h1 {
            font-size: clamp(2rem, 3.5vw, 3rem);
        }
        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] label {
            color: #193849;
        }
        [data-testid="stForm"] {
            background: rgba(232, 247, 245, 0.76);
            border: 1px solid rgba(25, 139, 129, 0.27);
            border-radius: 20px;
            padding: 24px;
        }
        [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(100deg, #098c76, #1177a8);
            color: white;
            border: none;
            font-weight: 700;
            border-radius: 12px;
            min-height: 2.9rem;
            padding: 0.7rem 1.5rem;
            box-shadow: 0 7px 18px rgba(2, 97, 115, 0.2);
        }
        [data-testid="stFormSubmitButton"] button:hover {
            color: white;
            border: none;
            background: linear-gradient(100deg, #057463, #115f90);
        }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                margin: 12px 9px 28px;
                padding: 1.4rem 1rem 1.8rem;
                border-radius: 16px;
            }
            [data-testid="stForm"] { padding: 12px; }
        }
        """.replace("BACKGROUND_PLACEHOLDER", background) + "</style>",
        unsafe_allow_html=True,
    )


style_page()


@st.cache_resource
def load_model_bundle():
    # Only load model files created by you or from a source you trust.
    with MODEL_FILE.open("rb") as file:
        bundle = pickle.load(file)
    required = {"model", "feature_columns", "class_labels"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Unexpected model bundle structure")
    return bundle


st.title("🇱🇰 Tourist Arrival Trend Predictor")
st.write(
    "Predict whether tourist arrivals from a selected country to Sri Lanka "
    "will **increase** or **not increase** in the following month."
)
st.caption(
    "Based on a Histogram-Based Gradient Boosting classifier trained on "
    "historical country-level monthly tourism data. Predictions are estimates, not guarantees."
)

try:
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error(
        "Model file not found. Place `tourist_trend_model.pkl` "
        "in the same folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(
        "The saved model could not be loaded. Check that this Python environment "
        "uses compatible versions of scikit-learn and its dependencies."
    )
    st.exception(exc)
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
labels = bundle["class_labels"]

country_columns = [col for col in feature_columns if col.startswith("Country_")]
countries = sorted(col[len("Country_"):] for col in country_columns)
if not countries:
    st.error("No country feature columns were found in the model bundle.")
    st.stop()

st.subheader("Enter monthly arrival data")
with st.form("trend_prediction_form"):
    country = st.selectbox("Country of origin", countries)
    year = int(st.number_input("Current month's year", min_value=2023, max_value=2030, value=2026, step=1))
    month = st.selectbox("Current month", MONTHS, index=4)

    st.write("**Tourist arrivals in four consecutive months**")
    current = int(st.number_input("Current month", min_value=0, value=0, step=1))
    previous_1 = int(st.number_input("Previous month", min_value=0, value=0, step=1))
    previous_2 = int(st.number_input("Two months ago", min_value=0, value=0, step=1))
    previous_3 = int(st.number_input("Three months ago", min_value=0, value=0, step=1))
    st.caption("Enter actual arrival counts. A value of 0 means zero arrivals, not missing data.")

    submitted = st.form_submit_button("Predict next-month trend", type="primary")

if submitted:
    month_num = MONTHS.index(month) + 1
    # The trained feature matrix contains a one-hot column for each known country.
    # Start with zeroes, then populate the same numeric features used in training.
    row = {column: 0 for column in feature_columns}
    values = {
        "Year": year,
        "MonthNum": month_num,
        "Arrivals": current,
        "Lag1_Arrivals": previous_1,
        "Lag2_Arrivals": previous_2,
        "Lag3_Arrivals": previous_3,
        "ChangeLastMonth": current - previous_1,
        "MeanLast3Months": (current + previous_1 + previous_2) / 3,
    }
    for name, value in values.items():
        if name in row:
            row[name] = value
    row[f"Country_{country}"] = 1

    sample = pd.DataFrame([row], columns=feature_columns)
    prediction = int(model.predict(sample)[0])
    class_ids = list(model.classes_)
    probabilities = model.predict_proba(sample)[0]
    increase_probability = float(probabilities[class_ids.index(1)])

    target_month = MONTHS[month_num % 12]
    target_year = year + (month_num == 12)

    st.subheader(f"Prediction for {target_month} {target_year}")
    predicted_label = labels.get(prediction, labels.get(str(prediction), str(prediction)))
    if prediction == 1:
        st.success(f"Predicted trend: **{predicted_label}**")
    else:
        st.info(f"Predicted trend: **{predicted_label}**")
    st.metric("Estimated probability of increase", f"{increase_probability:.1%}")
    st.progress(increase_probability)
    st.caption(
        "These are uncalibrated model probabilities. The model had limited performance "
        "on its 2026 holdout test set; results are for educational demonstration."
    )
