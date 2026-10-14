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
DATA_FILE = Path(__file__).with_name("Sri_Lanka_Tourist_Arrivals_2022_2026.csv")

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
    # Pickle is safe to open only when you trust who created the file.
    with MODEL_FILE.open("rb") as file:
        bundle = pickle.load(file)
    required = {"model", "feature_columns", "class_labels"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Unexpected saved model bundle")
    return bundle


@st.cache_data(show_spinner=False)
def load_arrival_history():
    """Replicate the notebook's cleaning and index months by country.

    Automatic mode only offers dates for which the current and three
    immediately preceding calendar months are all present in the CSV.
    """
    raw = pd.read_csv(DATA_FILE)
    required = {"Year", "Country", "Month", "Arrivals"}
    if not required.issubset(raw.columns):
        raise ValueError("Tourism CSV is missing expected columns")

    data = raw.copy()
    data["Arrivals"] = pd.to_numeric(data["Arrivals"], errors="coerce")
    data = data.dropna(subset=["Arrivals"]).copy()
    data = data.drop_duplicates(subset=["Year", "Country", "Month"])
    data = data[data["Year"] >= 2023].copy()
    mapping = {month: number for number, month in enumerate(MONTHS, start=1)}
    data["MonthNum"] = data["Month"].map(mapping)
    data = data.dropna(subset=["MonthNum"]).copy()
    data["PeriodIndex"] = data["Year"].astype(int) * 12 + data["MonthNum"].astype(int)

    lookup = {
        (str(record.Country), int(record.PeriodIndex)): int(record.Arrivals)
        for record in data[["Country", "PeriodIndex", "Arrivals"]].itertuples(index=False)
    }
    eligible = {}
    for country, country_rows in data.groupby("Country"):
        available = set(country_rows["PeriodIndex"].astype(int).tolist())
        good_months = sorted(
            period for period in available
            if all((period - lag) in available for lag in (1, 2, 3))
        )
        if good_months:
            eligible[str(country)] = good_months

    return lookup, eligible


def month_label(period_index):
    """Convert the continuous month index back to a readable month/year."""
    year, zero_month = divmod(int(period_index) - 1, 12)
    return f"{MONTHS[zero_month]} {year}"


def predict(model, expected_columns, country, year, month_number, arrivals):
    """Build the *exact* numeric features used during model training."""
    current, previous_1, previous_2, previous_3 = map(int, arrivals)
    row = {column: 0 for column in expected_columns}
    values = {
        "Year": int(year),
        "MonthNum": int(month_number),
        "Arrivals": current,
        "Lag1_Arrivals": previous_1,
        "Lag2_Arrivals": previous_2,
        "Lag3_Arrivals": previous_3,
        "ChangeLastMonth": current - previous_1,
        "MeanLast3Months": (current + previous_1 + previous_2) / 3,
    }
    for key, value in values.items():
        if key in row:
            row[key] = value

    country_column = f"Country_{country}"
    if country_column not in row:
        raise ValueError("This country was not represented during model training")
    row[country_column] = 1

    observation = pd.DataFrame([row], columns=expected_columns)
    category = int(model.predict(observation)[0])
    class_ids = list(model.classes_)
    proba = float(model.predict_proba(observation)[0][class_ids.index(1)])
    return category, proba


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
    st.error("Model file not found. Place `tourist_trend_model.pkl` next to `app.py`.")
    st.stop()
except Exception as exc:
    st.error("Could not load the model. Check the scikit-learn dependency version.")
    st.exception(exc)
    st.stop()

model = bundle["model"]
expected_columns = bundle["feature_columns"]
labels = bundle["class_labels"]
model_countries = sorted(
    col[len("Country_"):] for col in expected_columns
    if col.startswith("Country_")
)
if not model_countries:
    st.error("The trained model doesn't contain any country columns.")
    st.stop()

lookup = {}
eligible = {}
if DATA_FILE.is_file():
    try:
        lookup, eligible = load_arrival_history()
    except Exception as exc:
        st.warning("Historical arrivals could not be loaded. Manual entry is still available.")
        st.caption(str(exc))
else:
    st.warning("Historical CSV not found. Upload it to GitHub for automatic lookup.")

eligible_countries = sorted(set(model_countries).intersection(eligible))
mode_options = ["Automatic — use dataset", "Manual — enter arrivals"] if eligible_countries else ["Manual — enter arrivals"]

st.subheader("Select a country and month")
mode = st.radio(
    "Where should arrival figures come from?",
    options=mode_options,
    horizontal=True,
    help="Automatic mode looks up the current and preceding three months in the CSV. "
         "Manual mode is useful for newer months not in the dataset.",
)

if mode.startswith("Automatic"):
    countries = eligible_countries
else:
    countries = model_countries
preferred_country = "India" if "India" in countries else countries[0]
country = st.selectbox(
    "Country of origin",
    countries,
    index=countries.index(preferred_country),
)

if mode.startswith("Automatic"):
    period_options = eligible[country]
    years = sorted({(p - 1) // 12 for p in period_options}, reverse=True)
    year = int(st.selectbox("Current month's year", years))
    month_numbers = sorted(
        [((p - 1) % 12) + 1 for p in period_options if (p - 1) // 12 == year],
        reverse=True,
    )
    selected_month_num = int(st.selectbox(
        "Current month",
        month_numbers,
        format_func=lambda number: MONTHS[number - 1],
    ))
    period = year * 12 + selected_month_num
    four_counts = [lookup[(country, period - offset)] for offset in range(4)]

    st.info("The four consecutive months of arrival data will be filled automatically from the uploaded dataset.")
    with st.form("trend_prediction_form"):
        st.write("**Historical arrivals used by the model**")
        preview = pd.DataFrame({
            "Month": [month_label(period - lag) for lag in (0, 1, 2, 3)],
            "Tourist arrivals": four_counts,
        })
        st.dataframe(preview, hide_index=True, use_container_width=True)
        st.caption(
            "Source: uploaded Sri Lankan tourism CSV. Historical data are not updated live. "
            "Choose manual mode for newer observations."
        )
        submitted = st.form_submit_button("Predict next-month trend", type="primary")
else:
    year = int(st.number_input(
        "Current month's year", min_value=2023, max_value=2030, value=2026, step=1,
    ))
    selected_month_num = MONTHS.index(st.selectbox("Current month", MONTHS, index=4)) + 1

    with st.form("trend_prediction_form"):
        st.write("**Tourist arrivals in four consecutive months**")
        current = int(st.number_input("Current month", min_value=0, value=0, step=1))
        previous_1 = int(st.number_input("Previous month", min_value=0, value=0, step=1))
        previous_2 = int(st.number_input("Two months ago", min_value=0, value=0, step=1))
        previous_3 = int(st.number_input("Three months ago", min_value=0, value=0, step=1))
        four_counts = [current, previous_1, previous_2, previous_3]
        st.caption(
            "Enter actual tourist arrival counts. A zero means zero arrivals, not missing data. "
            "Use the months immediately preceding your selected month, including across year boundaries."
        )
        submitted = st.form_submit_button("Predict next-month trend", type="primary")

if submitted:
    try:
        category, prob_increase = predict(
            model, expected_columns, country, year, selected_month_num, four_counts
        )
    except Exception as exc:
        st.error("Unable to generate the prediction from these inputs.")
        st.exception(exc)
        st.stop()

    next_month_num = selected_month_num % 12 + 1
    next_year = year + (selected_month_num == 12)
    next_label = f"{MONTHS[next_month_num - 1]} {next_year}"
    predicted_label = labels.get(category, labels.get(str(category), str(category)))

    st.subheader(f"Prediction for {next_label}")
    if category == 1:
        st.success(f"Predicted trend: **{predicted_label}**")
    else:
        st.info(f"Predicted trend: **{predicted_label}**")
    st.metric("Estimated probability of increase", f"{prob_increase:.1%}")
    st.progress(prob_increase)
    st.caption(
        "These are uncalibrated model probabilities. The classifier showed limited performance "
        "on its 2026 held-out test set, so results are for educational demonstration only."
    )

