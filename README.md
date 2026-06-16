# 🏠 UK House Price Predictor

A machine learning project that predicts residential property sale prices across England and Wales, trained on **HM Land Registry 2025** transaction data.

Built as a capstone project for the **Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence**.

---

## 🌐 Live Demo

> Run the app locally following the instructions below.

![App Screenshot](https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=900&q=60)

---

## 📋 Project Overview

This project follows a full end-to-end ML workflow:

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_data_exploration.ipynb` | Data loading, cleaning, and visualisation |
| 2 | `02_feature_engineering.ipynb` | Feature creation, encoding, macroeconomic data |
| 3 | `03_modelling.ipynb` | Model training and comparison |
| 4 | `app.py` | Interactive Streamlit web application |

---

## 🔍 Features

**Property inputs:**
- Property type (Detached, Semi-detached, Terraced, Flat)
- New build or existing property
- Tenure (Freehold or Leasehold)
- County and town/city

**Engineered features:**
- Seasonality (month, quarter, season)
- Bank of England base rate (monthly, 2025)
- UK CPI inflation (monthly, 2025)
- Geographic target encoding (county and town median prices)

**Models trained:**
- Linear Regression (baseline)
- Random Forest
- XGBoost ✅ *(best performer — used in the app)*

---

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | £250,237 | £121,253 | 0.3787 |
| Random Forest | £230,841 | £106,690 | 0.4713 |
| **XGBoost** | £232,541 | £107,954 | 0.4634 |



---

## 🗂️ Repository Structure

```
uk-house-price-predictor/
│
├── 01_data_exploration.ipynb       # Phase 1: EDA and visualisation
├── 02_feature_engineering.ipynb    # Phase 2: Feature engineering
├── 03_modelling.ipynb              # Phase 3: ML models and evaluation
│
├── app.py                          # Phase 4: Streamlit web app
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes large data files
│
└── README.md                       # This file
```

> **Note:** Data files (`dataset_clean.csv`, `dataset_ml.csv`) and model files (`model_xgboost.pkl`, `features_list.pkl`) are excluded from this repository due to file size. See the **Getting Started** section below to reproduce them.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/uk-house-price-predictor.git
cd uk-house-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

1. Go to: https://www.gov.uk/government/collections/price-paid-data
2. Download the **2025 annual file**: `pp-2025.csv`
3. Place it in the root folder of the project

### 4. Run the notebooks in order

```
01_data_exploration.ipynb
02_feature_engineering.ipynb
03_modelling.ipynb
```

This will generate `dataset_clean.csv`, `dataset_ml.csv`, `model_xgboost.pkl`, and `features_list.pkl`.

### 5. Launch the web app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas | Data manipulation |
| Matplotlib | Visualisation |
| Scikit-learn | ML pipeline and baseline models |
| XGBoost | Primary predictive model |
| Streamlit | Interactive web application |

**External data sources:**
- [HM Land Registry Price Paid Data](https://www.gov.uk/government/collections/price-paid-data)
- [Bank of England Base Rate](https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate)
- [ONS CPI Inflation](https://www.ons.gov.uk/economy/inflationandpriceindices)

---

## 💡 Key Findings

- **Geographic location** is the strongest predictor of sale price, with county and town median prices accounting for the majority of model variance.
- **Property type** has a significant impact: detached properties command a substantial premium over flats.
- **Macroeconomic conditions** (BoE base rate and CPI) add meaningful signal, particularly for capturing seasonal market dynamics.
- **XGBoost** outperforms Linear Regression and Random Forest on both RMSE and R², confirming gradient boosting as the preferred approach for tabular property data.

---

## 🔮 Potential Improvements

- Add postcode-level features using Ordnance Survey data
- Incorporate number of bedrooms and floor area (EPC dataset)
- Expand to multi-year training data for temporal trend modelling
- Deploy the app publicly via Streamlit Cloud

---

## 👤 Author

**Riccardo Prenza**  
Professional Certificate in Machine Learning and Artificial Intelligence — Imperial College London  
[LinkedIn](https://www.linkedin.com/in/riccardoprenza/) · [GitHub](https://github.com/rprenza)

---

## 📄 Licence

This project is licensed under the MIT Licence. The underlying Land Registry data is published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
