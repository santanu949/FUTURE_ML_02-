import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from src.models.evaluation import plot_confusion_matrix, plot_roc_curve, plot_lift_chart, plot_profit_curve

st.set_page_config(page_title="AI Churn Intelligence", layout="wide", page_icon="📡")

# Custom CSS for modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

st.title("📡 AI-Driven Telecom Churn Intelligence Platform")

# 1. Sidebar - Configuration & Data Loading
with st.sidebar:
    st.header("⚙️ Configuration")
    model_version = st.selectbox("Model Version", ["v3 - Ensemble Stacking", "v2 - XGBoost"])
    
    st.divider()
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")
    
    if st.button("🔄 Train New Model"):
        st.warning("Training started... this may take 1-2 minutes.")
        from src.models.train import train_ensemble_pipeline
        train_ensemble_pipeline()
        st.success("Model trained and saved!")

# Load Data
DATA_PATH = "data/raw/telecom_churn_v3.csv"
MODEL_PATH = "models/churn_ensemble_v3.pkl"

if os.path.exists(DATA_PATH) or uploaded_file:
    df = pd.read_csv(uploaded_file if uploaded_file else DATA_PATH)
    
    # 2. Executive Summary Metrics
    st.subheader("📊 Business Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    churn_rate = df['churn'].mean()
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{churn_rate:.2%}")
    c3.metric("Avg Tenure", f"{df['tenure_days'].mean()/365:.1f} Yrs")
    c4.metric("Avg Monthly Revenue", f"${df['estimated_salary'].mean()/120:.2f}")

    # 3. Intelligence Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Insights", "🧪 Model Performance", "🔮 Real-time Prediction"])

    with tab1:
        st.header("Customer Behavioral Insights")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("### Usage Drop Ratio vs Churn")
            fig, ax = plt.subplots()
            df.boxplot(column='usage_drop_ratio', by='churn', ax=ax)
            st.pyplot(fig)
            
        with col_right:
            st.write("### Churn by Plan Type")
            plan_churn = df.groupby('plan_type')['churn'].mean().reset_index()
            fig, ax = plt.subplots()
            ax.bar(plan_churn['plan_type'], plan_churn['churn'], color=['#ff9999','#66b3ff'])
            st.pyplot(fig)

    with tab2:
        st.header("Model Evaluation & Business Impact")
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # Predict for validation visuals (Sample)
            test_df = df.sample(min(1000, len(df)))
            y_true = test_df['churn']
            X_test = test_df.drop(['customer_id', 'churn'], axis=1)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            l1, r1 = st.columns(2)
            with l1:
                st.pyplot(plot_confusion_matrix(y_true, y_pred))
            with r1:
                st.pyplot(plot_roc_curve(y_true, y_prob))
                
            l2, r2 = st.columns(2)
            with l2:
                st.pyplot(plot_lift_chart(y_true, y_prob))
            with r2:
                st.pyplot(plot_profit_curve(y_true, y_prob))
        else:
            st.error("No production model found. Please run training from the sidebar.")

    with tab3:
        st.header("Individual Customer Risk Assessment")
        with st.form("predict_form"):
            f1, f2, f3 = st.columns(3)
            age = f1.slider("Age", 18, 90, 35)
            salary = f2.number_input("Estimated Salary", 10000, 300000, 50000)
            tenure = f3.slider("Tenure (Days)", 0, 2000, 365)
            
            f4, f5, f6 = st.columns(3)
            complaints = f4.number_input("Total Complaints", 0, 10, 0)
            usage_drop = f5.slider("Usage Drop Ratio", 0.0, 2.0, 1.0)
            plan = f6.selectbox("Plan Type", ["Prepaid", "Postpaid"])
            
            submit = st.form_submit_button("Assess Risk")
            
            if submit and os.path.exists(MODEL_PATH):
                # Build dummy row
                input_data = df.iloc[0:1].copy()
                input_data['age'] = age
                input_data['estimated_salary'] = salary
                input_data['tenure_days'] = tenure
                input_data['total_complaints'] = complaints
                input_data['usage_drop_ratio'] = usage_drop
                input_data['plan_type'] = plan
                
                prob = model.predict_proba(input_data.drop(['customer_id', 'churn'], axis=1))[0, 1]
                
                st.divider()
                st.write(f"## Churn Risk: {prob:.1%}")
                if prob > 0.6:
                    st.error("🚨 HIGH RISK CUSTOMER")
                    st.write("### Recommended Retention Strategy:")
                    st.info("Offer 20% loyalty discount for 6 months + Direct Support Call.")
                else:
                    st.success("✅ LOW RISK CUSTOMER")
                    st.write("### Recommended Retention Strategy:")
                    st.write("Standard maintenance - monitor usage patterns monthly.")

else:
    st.info("👋 Welcome! Please upload a dataset or run 'python run.py generate' to start.")
