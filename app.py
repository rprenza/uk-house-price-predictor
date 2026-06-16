import streamlit as st
import pandas as pd
import numpy as np
import pickle

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


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features_list.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_geo_data():
    df = pd.read_csv('dataset_ml.csv')
    county_median = df.groupby('county')['price'].median() if 'county' in df.columns else None
    town_median   = df.groupby('town')['price'].median()   if 'town'   in df.columns else None
    if county_median is None or town_median is None:
        df2 = pd.read_csv('dataset_clean.csv')
        county_median = df2.groupby('county')['price'].median()
        town_median   = df2.groupby('town')['price'].median()
    return county_median, town_median

@st.cache_data
def load_town_county_map():
    try:
        df = pd.read_csv('dataset_clean.csv', usecols=['county','town'])
        return df.dropna().drop_duplicates()
    except Exception:
        return None


# ── Macro data 2025 ───────────────────────────────────────────────────────────
BOE_RATES = {
    1:4.75,2:4.50,3:4.50,4:4.25,5:4.25,6:4.25,
    7:4.25,8:4.25,9:4.00,10:4.00,11:4.00,12:4.00
}
INFLATION = {
    1:3.0,2:2.8,3:2.6,4:3.5,5:3.4,6:3.3,
    7:3.2,8:3.1,9:3.0,10:3.0,11:2.9,12:2.8
}

PROPERTY_LABELS = {
    'D': 'Detached',
    'S': 'Semi-detached',
    'T': 'Terraced',
    'F': 'Flat / Maisonette'
}

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

SEASON_MAP = {
    **{m: (0,'Winter') for m in [12,1,2]},
    **{m: (1,'Spring') for m in [3,4,5]},
    **{m: (2,'Summer') for m in [6,7,8]},
    **{m: (3,'Autumn') for m in [9,10,11]}
}


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 UK House Price Predictor")
st.markdown(
    "Estimate the sale price of a UK residential property using an **XGBoost** model "
    "trained on **HM Land Registry 2025** transaction data."
)
st.divider()

# ── Load assets ───────────────────────────────────────────────────────────────
try:
    model, features = load_model()
    county_median, town_median = load_geo_data()
    town_county_map = load_town_county_map()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Required file not found: **{e}**\n\n"
        "Make sure the following files are in the same folder as `app.py`:\n"
        "- `model_xgboost.pkl`\n"
        "- `features_list.pkl`\n"
        "- `dataset_ml.csv`"
    )
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Property Details")
    st.markdown("Fill in the details and click **Predict Price**.")
    st.divider()

    # Property type
    prop_type = st.selectbox(
        "Property Type",
        options=list(PROPERTY_LABELS.keys()),
        format_func=lambda x: PROPERTY_LABELS[x]
    )

    # New build
    new_build = st.radio("New Build?", ["No", "Yes"], horizontal=True)
    is_new_build = 1 if new_build == "Yes" else 0

    # Tenure
    tenure = st.radio("Tenure", ["Freehold", "Leasehold"], horizontal=True)
    is_freehold = 1 if tenure == "Freehold" else 0

    st.divider()
    st.subheader("📍 Location")

    counties = sorted(county_median.index.tolist())
    county = st.selectbox("County", options=counties)

    # Filter towns by selected county
    if town_county_map is not None:
        towns_in_county = sorted(
            town_county_map[town_county_map['county'] == county]['town']
            .dropna().unique().tolist()
        )
    else:
        towns_in_county = sorted(town_median.index.tolist())

    town = st.selectbox("Town / City", options=towns_in_county)

    st.divider()
    st.subheader("📅 Sale Date")
    month = st.slider("Month", min_value=1, max_value=12, value=6)
    quarter = (month - 1) // 3 + 1
    season_num, season_name = SEASON_MAP[month]
    st.caption(f"**{MONTH_NAMES[month-1]}** · Q{quarter} · {season_name}")

    st.divider()
    predict_btn = st.button("🔍 Predict Price", use_container_width=True)


# ── Main panel ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # Resolve geographic medians
        c_med = float(county_median.get(county, county_median.median()))
        t_med = float(town_median.get(town,   town_median.median()))

        boe  = BOE_RATES.get(month, 4.25)
        infl = INFLATION.get(month, 3.0)

        # One-hot encoding (drop_first=True dropped 'D' as reference)
        type_F = 1 if prop_type == 'F' else 0
        type_S = 1 if prop_type == 'S' else 0
        type_T = 1 if prop_type == 'T' else 0

        row = {
            'month':        month,
            'quarter':      quarter,
            'season_num':   season_num,
            'boe_rate':     boe,
            'cpi':          infl,
            'is_new_build': is_new_build,
            'is_freehold':  is_freehold,
            'county_median': c_med,
            'town_median':   t_med,
            'type_F':        type_F,
            'type_S':        type_S,
            'type_T':        type_T,
        }

        # Align to trained feature order
        X_input = pd.DataFrame([{f: row.get(f, 0) for f in features}])

        prediction = float(model.predict(X_input)[0])
        prediction = max(10_000, prediction)
        low  = prediction * 0.90
        high = prediction * 1.10

        # Prediction card
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

        # Input summary
        st.subheader("📋 Input Summary")
        summary = {
            "Property Type": PROPERTY_LABELS[prop_type],
            "New Build":     new_build,
            "Tenure":        tenure,
            "County":        county,
            "Town":          town,
            "Month":         MONTH_NAMES[month - 1],
            "Quarter":       f"Q{quarter}",
            "Season":        season_name,
            "BoE Base Rate": f"{boe}%",
            "CPI Inflation": f"{infl}%",
            "County Median": f"£{c_med:,.0f}",
            "Town Median":   f"£{t_med:,.0f}",
        }
        st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))

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

    # County median card
    if county in county_median.index:
        c_val = county_median[county]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{c_val:,.0f}</div>
            <div class="metric-label">Median price in {county}</div>
        </div>
        """, unsafe_allow_html=True)

    # Town median card
    if town in town_median.index:
        t_val = town_median[town]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{t_val:,.0f}</div>
            <div class="metric-label">Median price in {town}</div>
        </div>
        """, unsafe_allow_html=True)

    # BoE rate card
    boe_display = BOE_RATES.get(month, 4.25)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{boe_display}%</div>
        <div class="metric-label">BoE Base Rate · {MONTH_NAMES[month-1]} 2025</div>
    </div>
    """, unsafe_allow_html=True)

    # Top 10 counties chart
    st.subheader("🗺️ Top 10 Counties by Median Price")
    top10 = county_median.sort_values(ascending=False).head(10)
    chart_df = pd.DataFrame({'County': top10.index, 'Median Price (£)': top10.values})
    st.bar_chart(chart_df.set_index('County'))


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>**Data source:** HM Land Registry Price Paid Data 2025 &nbsp;·&nbsp; "
    "**Model:** XGBoost Regressor &nbsp;·&nbsp; "
    "**Author:** Riccardo Prenza &nbsp;·&nbsp; "
    "Built with Python & Streamlit</small>",
    unsafe_allow_html=True
)
