import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1a5276;
        color: white;
        border-radius: 8px;
        padding: 0.6em 2em;
        font-size: 1.1em;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #2e86c1; }
    .prediction-box {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        border-radius: 16px;
        padding: 2em;
        text-align: center;
        color: white;
        margin: 1em 0;
    }
    .prediction-title { font-size: 1.1em; opacity: 0.85; margin-bottom: 0.3em; }
    .prediction-price { font-size: 3em; font-weight: 800; letter-spacing: -1px; }
    .prediction-range { font-size: 0.95em; opacity: 0.75; margin-top: 0.5em; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5em;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1em;
    }
    .metric-value { font-size: 2em; font-weight: 700; color: #1a5276; }
    .metric-label { font-size: 0.9em; color: #666; margin-top: 4px; }
    .info-box {
        background: #eaf4fb;
        border-left: 4px solid #2e86c1;
        border-radius: 6px;
        padding: 1em 1.2em;
        margin: 1em 0;
        font-size: 0.92em;
        color: #1a5276;
    }
    h1 { color: #1a5276; }
    h3 { color: #1a5276; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BOE_RATES = {
    1:4.75,2:4.50,3:4.50,4:4.25,5:4.25,6:4.25,
    7:4.25,8:4.25,9:4.00,10:4.00,11:4.00,12:4.00
}
INFLATION = {
    1:3.0,2:2.8,3:2.6,4:3.5,5:3.4,6:3.3,
    7:3.2,8:3.1,9:3.0,10:3.0,11:2.9,12:2.8
}
PROPERTY_LABELS = {
    "D":"Detached","S":"Semi-detached","T":"Terraced","F":"Flat / Maisonette"
}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
SEASON_MAP = {
    **{m:(0,"Winter") for m in [12,1,2]},
    **{m:(1,"Spring") for m in [3,4,5]},
    **{m:(2,"Summer") for m in [6,7,8]},
    **{m:(3,"Autumn") for m in [9,10,11]}
}


# ── Load geo data from CSV ────────────────────────────────────────────────────
@st.cache_data
def load_geo():
    county_df    = pd.read_csv("geo_county.csv")
    town_df      = pd.read_csv("geo_town.csv")
    town_county  = pd.read_csv("geo_town_county.csv")
    county_median = county_df.set_index("county")["county_median"]
    town_median   = town_df.set_index("town")["town_median"]
    return county_median, town_median, town_county


# ── Train model on startup ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model — please wait (~1 min)...")
def train_model():
    df = pd.read_csv("dataset_ml_small.csv")
    target   = "price"
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model, features


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 UK House Price Predictor")
st.markdown(
    "Estimate the sale price of a UK residential property using an **XGBoost** model "
    "trained on **HM Land Registry 2025** transaction data."
)
st.divider()

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    county_median, town_median, town_county = load_geo()
    model, features = train_model()
except FileNotFoundError as e:
    st.error(f"⚠️ Required file not found: **{e}**")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Property Details")
    st.markdown("Fill in the details and click **Predict Price**.")
    st.divider()

    prop_type    = st.selectbox(
        "Property Type",
        options=list(PROPERTY_LABELS.keys()),
        format_func=lambda x: PROPERTY_LABELS[x]
    )
    new_build    = st.radio("New Build?", ["No","Yes"], horizontal=True)
    tenure       = st.radio("Tenure", ["Freehold","Leasehold"], horizontal=True)
    is_new_build = 1 if new_build == "Yes" else 0
    is_freehold  = 1 if tenure == "Freehold" else 0

    st.divider()
    st.subheader("📍 Location")
    counties = sorted(county_median.index.tolist())
    county   = st.selectbox("County", options=counties)

    towns_in_county = sorted(
        town_county[town_county["county"] == county]["town"]
        .dropna().unique().tolist()
    )
    town = st.selectbox("Town / City", options=towns_in_county)

    st.divider()
    st.subheader("📅 Sale Date")
    month       = st.slider("Month", min_value=1, max_value=12, value=6)
    quarter     = (month - 1) // 3 + 1
    season_num, season_name = SEASON_MAP[month]
    st.caption(f"**{MONTH_NAMES[month-1]}** · Q{quarter} · {season_name}")

    st.divider()
    predict_btn = st.button("🔍 Predict Price", use_container_width=True)


# ── Main panel ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        c_med = float(county_median.get(county, county_median.median()))
        t_med = float(town_median.get(town,     town_median.median()))
        boe   = BOE_RATES.get(month, 4.25)
        infl  = INFLATION.get(month, 3.0)

        type_F = 1 if prop_type == "F" else 0
        type_S = 1 if prop_type == "S" else 0
        type_T = 1 if prop_type == "T" else 0

        row = {
            "month": month, "quarter": quarter, "season_num": season_num,
            "boe_rate": boe, "cpi": infl,
            "is_new_build": is_new_build, "is_freehold": is_freehold,
            "county_median": c_med, "town_median": t_med,
            "type_F": type_F, "type_S": type_S, "type_T": type_T,
        }
        X_input    = pd.DataFrame([{f: row.get(f, 0) for f in features}])
        prediction = float(model.predict(X_input)[0])
        prediction = max(10_000, prediction)
        low, high  = prediction * 0.90, prediction * 1.10

        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-title">Estimated Sale Price</div>
            <div class="prediction-price">£{prediction:,.0f}</div>
            <div class="prediction-range">Indicative range &nbsp;·&nbsp; £{low:,.0f} – £{high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            ℹ️ This estimate is based on 2025 Land Registry transaction data.
            The ±10% range reflects typical model uncertainty for individual properties.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Input Summary")
        summary = {
            "Property Type": PROPERTY_LABELS[prop_type],
            "New Build": new_build, "Tenure": tenure,
            "County": county, "Town": town,
            "Month": MONTH_NAMES[month-1], "Quarter": f"Q{quarter}",
            "Season": season_name,
            "BoE Base Rate": f"{boe}%", "CPI Inflation": f"{infl}%",
            "County Median": f"£{c_med:,.0f}", "Town Median": f"£{t_med:,.0f}",
        }
        st.table(pd.DataFrame(summary.items(), columns=["Parameter","Value"]))

    else:
        st.markdown("""
        <div class="info-box">
            👈 Fill in the property details in the sidebar, then click <strong>Predict Price</strong>.
        </div>
        """, unsafe_allow_html=True)
        st.image(
            "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=700&q=60",
            caption="UK residential property market",
            use_container_width=True
        )

with col_right:
    st.subheader("📈 Market Context")

    if county in county_median.index:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{county_median[county]:,.0f}</div>
            <div class="metric-label">Median price in {county}</div>
        </div>""", unsafe_allow_html=True)

    if town in town_median.index:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{town_median[town]:,.0f}</div>
            <div class="metric-label">Median price in {town}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{BOE_RATES.get(month,4.25)}%</div>
        <div class="metric-label">BoE Base Rate · {MONTH_NAMES[month-1]} 2025</div>
    </div>""", unsafe_allow_html=True)

    st.subheader("🗺️ Top 10 Counties by Median Price")
    top10 = county_median.sort_values(ascending=False).head(10)
    st.bar_chart(pd.DataFrame({"Median Price (£)": top10}))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>**Data source:** HM Land Registry Price Paid Data 2025 &nbsp;·&nbsp; "
    "**Model:** XGBoost Regressor &nbsp;·&nbsp; "
    "**Author:** Riccardo Prenza &nbsp;·&nbsp; "
    "Built with Python & Streamlit</small>",
    unsafe_allow_html=True
)
