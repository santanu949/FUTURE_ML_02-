# 📡 Telecom Churn Intelligence: Production-Grade ML System

## 📌 Project Overview
This project is an advanced, production-ready machine learning system designed to predict and mitigate customer churn in the telecom industry. Moving beyond simple classification, it incorporates temporal intelligence, business ROI simulation, and a full-stack deployment architecture.

## 🚀 Key Innovations
- **Temporal Intelligence**: Features engineered from 6-month usage trends, rolling averages, and tenure buckets.
- **Advanced Modeling**: Gradient Boosting (XGBoost) with hyperparameter tuning and cross-validation.
- **Business-First Evaluation**: Custom scoring metrics that penalize False Negatives (missing churners) based on actual revenue loss.
- **Decision Layer**: Automated retention strategy assignment (Discounts, Support Calls) with ROI simulation.
- **Production Architecture**: Modular code structure, FastAPI service, and Streamlit monitoring dashboard.

---

## 📂 Project Structure
```text
FUTURE_ML_02--main/
├── src/
│   ├── data/           # Realistic data generation & validation
│   ├── features/       # Scikit-learn pipelines & feature engineering
│   ├── models/         # XGBoost training, tuning, & strategy simulation
│   ├── api/            # FastAPI real-time prediction service
│   └── app/            # Streamlit intelligence dashboard
├── tests/              # Unit tests for core logic
├── models/             # Serialized production models (.pkl)
├── data/               # Raw and processed datasets
├── run.py              # Unified CLI entry point
└── README.md           # This documentation
```

---

## 🛠️ Tech Stack
- **Core**: Python 3.13, Pandas, NumPy
- **ML**: XGBoost, Scikit-learn, Imbalanced-learn (SMOTE)
- **Deployment**: FastAPI, Uvicorn, Streamlit
- **Quality**: Pytest, Pydantic, Logging

---

## ⚙️ Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements_v2.txt
```

### 2. Unified CLI Usage
The project uses a single entry point `run.py` to manage all tasks:

- **Generate Realistic Data**:
  ```bash
  python run.py generate
  ```
- **Train Production Model**:
  ```bash
  python run.py train
  ```
- **Launch Prediction API**:
  ```bash
  python run.py api
  ```
- **Launch Dashboard**:
  ```bash
  python run.py app
  ```

---

## 🔄 Production Pipeline & Data Flow
1. **Data Ingestion**: Realistic simulation of customer behavior including usage drops and complaint spikes.
2. **Feature Pipeline**: Custom Transformers handle tenure grouping and behavioral interaction features (e.g., Data/Call ratio).
3. **Training & Tuning**: `RandomizedSearchCV` optimizes XGBoost parameters across stratified cross-validation folds.
4. **Strategy Simulation**: The model assigns specific interventions (High Discount vs. Standard Offer) based on churn probability.
5. **Real-time Serving**: FastAPI provides a `/predict` endpoint for CRM integration.

---

## 📊 Business Impact & ROI
The system evaluates success not just by Accuracy, but by **Net Profit Saved**. By simulating a retention strategy where a $1000 CLV (Customer Lifetime Value) is protected through targeted discounts, the system provides a projected ROI for marketing teams.

---

## 🧪 Testing
Run unit tests to ensure pipeline consistency:
```bash
pytest tests/
```

---
**Author**: Santanu
**Internship**: Future Interns (Advanced ML Task 2)
