# 📡 AI-Powered Telecom Churn Intelligence: Enterprise Solution

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-Ensemble%20Stacking-green.svg)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

## 📌 Project Overview
This project delivers a production-grade **Autonomous Decision Intelligence System (ADIS)** for predicting and mitigating customer churn in the telecommunications sector. Unlike standard ML models, this system bridges the gap between prediction and business action by incorporating a **Retention Strategy Engine** and an interactive **Intelligence Dashboard**.

The solution leverages an **Ensemble Stacking Architecture** (XGBoost + LightGBM + Random Forest) to achieve superior predictive accuracy on high-dimensional, behavioral telecom data.

## 🚀 50+ Key Enhancements & Features

### 1. Advanced Feature Engineering (20+ Features)
- **Temporal Dynamics**: Tenure days converted to life-stage buckets (New, Established, Senior, Legacy).
- **Behavioral Ratios**: Data-per-call, SMS-to-call, and usage-to-salary indices.
- **Financial Segmentation**: ARPU (Average Revenue Per User) estimation and High-Value Customer (HVC) tagging.
- **Stability Metrics**: Recharge consistency and usage stability indices.
- **Risk Indicators**: Complaint-to-tenure ratio and usage drop volatility.

### 2. State-of-the-Art Modeling
- **Ensemble Stacking**: Combines XGBoost, LightGBM, and Random Forest with a Logistic Meta-Learner.
- **Robust Preprocessing**: Scikit-learn pipelines with Robust Scaling and Iterative Imputation.
- **Advanced Optimization**: Stratified 5-Fold Cross-Validation and Randomized Search for hyperparameter tuning.

### 3. Business Intelligence Layer
- **ROI Simulation**: Real-time calculation of potential net profit based on saved Customer Lifetime Value (CLV).
- **Retention Strategy Engine**: Automated intervention suggestions (e.g., "Immediate Concierge Support" for high-risk segments).
- **Profit Curves**: Visualizing the optimal decision threshold for maximum business profitability.

### 4. Interactive Enterprise UI
- **Live ROI Calculator**: See the financial impact of your retention policies.
- **Performance Tab**: Gain/Lift charts, ROC-AUC, and Confusion Matrix heatmaps.
- **Real-time Assessment**: Predict risk for individual customers through an intuitive interface.

---

## 📂 System Architecture & Modules

```text
FUTURE_ML_02--main/
├── src/
│   ├── data/           # Enhanced data generation & realistic noise simulation
│   ├── features/       # Advanced Feature Engineering (AFE) pipelines
│   ├── models/         # Ensemble Stacking, Cross-Validation, & Evaluation
│   ├── api/            # FastAPI real-time prediction service
│   └── app/            # Streamlit intelligence dashboard
├── data/               # Versioned raw and processed datasets
├── models/             # Serialized production-ready models (.pkl)
├── tests/              # Unit tests for pipeline validation
├── run.py              # Centralised CLI for the entire system
└── README.md           # Professional project documentation
```

---

## ⚙️ Setup & Execution Guide

### 1. Prerequisites
- Python 3.13+
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/santanu949/FUTURE_ML_02-.git
cd FUTURE_ML_02--main

# Install dependencies
pip install -r requirements_v2.txt
pip install xgboost lightgbm catboost optuna shap
```

### 3. Running the System
The project uses a unified CLI `run.py` to handle all workflows:

| Task | Command |
| :--- | :--- |
| **Data Generation** | `python run.py generate` |
| **Model Training** | `python run.py train` |
| **Intelligence UI** | `python run.py app` |
| **Production API** | `python run.py api` |

---

## 🔄 Workflow & Data Pipeline
1. **Ingestion**: `generator.py` creates 10,000+ realistic records with missing values and noise.
2. **AFE (Advanced Feature Engineering)**: `build_features.py` transforms raw stats into meaningful behavioral signals.
3. **Training**: `train.py` executes the stacking ensemble with 5-fold cross-validation.
4. **Strategy**: The model outputs are passed to the retention engine to suggest business actions.
5. **Visualization**: The Streamlit app provides an executive-level summary of all findings.

---

## 📊 Business Logic
The model is tuned to minimize **False Negatives** (customers who leave without being detected). 
- **Cost of Miss**: $450
- **Cost of False Alarm**: $40
- **Retention Saving**: $300 (Net)

---
**Author**: Santanu
**Internship**: Future Interns (Machine Learning Task 2)
**Status**: Production Ready 🚀
