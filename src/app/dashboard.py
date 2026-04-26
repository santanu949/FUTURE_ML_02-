import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("📡 Telecom Churn Intelligence Dashboard")

# Load data and model
DATA_PATH = "data/raw/telecom_churn_v2.csv"
MODEL_PATH = "models/churn_model_v2.pkl"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    state_filter = st.sidebar.multiselect("Select States", options=df['state'].unique(), default=df['state'].unique())
    df_filtered = df[df['state'].isin(state_filter)]

    # Layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df_filtered))
    col2.metric("Churn Rate", f"{(df_filtered['churn'].mean()*100):.2f}%")
    col3.metric("Avg Revenue (Salary)", f"${df_filtered['estimated_salary'].mean():,.0f}")

    # Visuals
    st.subheader("Churn Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(data=df_filtered, x='tenure_months', hue='churn', multiple='stack', ax=ax[0])
    ax[0].set_title("Tenure vs Churn")
    
    sns.boxplot(data=df_filtered, x='churn', y='usage_drop_ratio', ax=ax[1])
    ax[1].set_title("Usage Drop Ratio vs Churn")
    
    st.pyplot(fig)

    # Model Interaction
    if os.path.exists(MODEL_PATH):
        st.divider()
        st.subheader("🔮 Real-time Prediction")
        
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 80, 30)
        salary = c2.number_input("Estimated Salary", 20000, 300000, 50000)
        tenure = c3.slider("Tenure (Months)", 0, 60, 12)
        
        c4, c5, c6 = st.columns(3)
        complaints = c4.number_input("Total Complaints", 0, 10, 0)
        usage_drop = c5.slider("Usage Drop Ratio", 0.0, 2.0, 1.0)
        plan = c6.selectbox("Plan Type", ["Prepaid", "Postpaid"])
        
        # Simple prediction button
        if st.button("Predict Risk"):
            model = joblib.load(MODEL_PATH)
            # Create a full dummy row for the pipeline
            dummy_data = df.iloc[0:1].copy()
            dummy_data['age'] = age
            dummy_data['estimated_salary'] = salary
            dummy_data['tenure_months'] = tenure
            dummy_data['total_complaints'] = complaints
            dummy_data['usage_drop_ratio'] = usage_drop
            dummy_data['plan_type'] = plan
            
            prob = model.predict_proba(dummy_data.drop(['customer_id', 'churn'], axis=1))[0, 1]
            st.write(f"### Churn Probability: {prob:.2%}")
            
            if prob > 0.5:
                st.error("⚠️ HIGH RISK OF CHURN")
            else:
                st.success("✅ LOW RISK")

else:
    st.error("Data not found. Please run the training pipeline first.")
