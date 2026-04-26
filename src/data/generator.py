import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TelecomDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.partners = ['Reliance Jio', 'Vodafone', 'BSNL', 'Airtel']
        self.states = ['Karnataka', 'Mizoram', 'Arunachal Pradesh', 'Tamil Nadu', 'Tripura', 'West Bengal', 'Maharashtra', 'Delhi']
        self.cities = ['Kolkata', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore']
        self.plans = ['Prepaid', 'Postpaid']

    def generate(self, num_customers=5000, months=6):
        print(f"Generating realistic data for {num_customers} customers over {months} months...")
        
        customers = []
        history = []

        start_date = datetime(2023, 1, 1)

        for i in range(num_customers):
            # Customer Static Info
            cust_id = i + 1
            gender = np.random.choice(['M', 'F'])
            age = np.random.randint(18, 75)
            state = np.random.choice(self.states)
            city = np.random.choice(self.cities)
            salary = np.random.randint(20000, 250000)
            plan_type = np.random.choice(self.plans, p=[0.7, 0.3])
            
            # Registration date (tenure)
            registration_date = start_date - timedelta(days=np.random.randint(30, 1000))
            tenure_months = (start_date - registration_date).days // 30
            
            # Churn Probability Factors
            # Higher churn if: low tenure, high complaints, sudden usage drop
            base_churn_prob = 0.05
            if tenure_months < 6: base_churn_prob += 0.1
            
            # Monthly usage simulation
            total_calls = 0
            total_data = 0
            complaints = np.random.poisson(0.1) # average 0.1 complaints
            
            usage_trend = []
            for m in range(months):
                # Monthly stats
                m_calls = np.random.randint(10, 300)
                m_data = np.random.randint(500, 20000) # MB
                m_sms = np.random.randint(0, 100)
                m_recharges = np.random.randint(1, 5) if plan_type == 'Prepaid' else 1
                
                # Introduce a "Drop" in usage for some users
                is_dropping = (i % 10 == 0) and (m >= months - 2)
                if is_dropping:
                    m_calls //= 4
                    m_data //= 4
                    base_churn_prob += 0.2
                
                usage_trend.append({
                    'month': m + 1,
                    'calls': m_calls,
                    'data_mb': m_data,
                    'sms': m_sms,
                    'recharges': m_recharges
                })
                
            # Final Churn Label
            churn = 1 if np.random.random() < min(base_churn_prob, 0.95) else 0
            
            # Aggregate stats for the "Snapshot" dataset
            last_month = usage_trend[-1]
            prev_month = usage_trend[-2]
            
            customers.append({
                'customer_id': cust_id,
                'gender': gender,
                'age': age,
                'state': state,
                'city': city,
                'estimated_salary': salary,
                'plan_type': plan_type,
                'tenure_months': tenure_months,
                'total_complaints': complaints,
                'avg_calls_3m': np.mean([u['calls'] for u in usage_trend[-3:]]),
                'avg_data_3m': np.mean([u['data_mb'] for u in usage_trend[-3:]]),
                'usage_drop_ratio': (last_month['calls'] + 1) / (prev_month['calls'] + 1),
                'recharge_frequency': np.mean([u['recharges'] for u in usage_trend]),
                'last_month_calls': last_month['calls'],
                'last_month_data': last_month['data_mb'],
                'churn': churn
            })

        df = pd.DataFrame(customers)
        
        # Add realistic "messiness"
        # 1. Missing values
        mask = np.random.random(df.shape) < 0.02
        df = df.mask(mask & (df.columns != 'churn') & (df.columns != 'customer_id'))
        
        # 2. Outliers
        df.loc[df.sample(frac=0.01).index, 'estimated_salary'] *= 10
        
        return df

if __name__ == "__main__":
    gen = TelecomDataGenerator()
    data = gen.generate(num_customers=5000)
    os.makedirs('data/raw', exist_ok=True)
    data.to_csv('data/raw/telecom_churn_v2.csv', index=False)
    print("Saved to data/raw/telecom_churn_v2.csv")
