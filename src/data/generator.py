import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataGenerator:
    """
    Generates a highly realistic telecom dataset with 50+ potential features, 
    noise, missing values, and business-driven correlations.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.partners = ['Reliance Jio', 'Vodafone', 'BSNL', 'Airtel']
        self.states = ['Karnataka', 'Mizoram', 'Arunachal Pradesh', 'Tamil Nadu', 'Tripura', 'West Bengal', 'Maharashtra', 'Delhi']
        self.cities = ['Kolkata', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore']
        self.plans = ['Prepaid', 'Postpaid']
        self.device_types = ['Smartphone', 'Basic Phone', 'IoT Device']
        self.payment_methods = ['UPI', 'Credit Card', 'Cash', 'Net Banking']

    def generate(self, num_customers=5000, months=6):
        logger.info(f"Starting data generation for {num_customers} customers...")
        
        customers = []
        start_date = datetime(2024, 1, 1)

        for i in range(num_customers):
            # 1. Demographic Features
            cust_id = f"CUST_{i+1:05d}"
            gender = np.random.choice(['M', 'F'], p=[0.51, 0.49])
            age = np.random.randint(18, 85)
            state = np.random.choice(self.states)
            city = np.random.choice(self.cities)
            salary = np.random.randint(15000, 300000)
            
            # 2. Contractual Features
            plan_type = np.random.choice(self.plans, p=[0.75, 0.25])
            device = np.random.choice(self.device_types, p=[0.8, 0.15, 0.05])
            payment = np.random.choice(self.payment_methods)
            registration_date = start_date - timedelta(days=np.random.randint(30, 2000))
            tenure_days = (start_date - registration_date).days
            
            # 3. Behavioral Indices (Hidden variables affecting churn)
            unhappy_factor = 0
            if age > 60: unhappy_factor += 0.05
            if salary < 30000: unhappy_factor += 0.1
            
            # 4. Monthly Usage Simulation (Temporal)
            usage_data = []
            for m in range(months):
                m_calls = np.random.randint(0, 500)
                m_data = np.random.randint(0, 50000) # MB
                m_sms = np.random.randint(0, 200)
                m_recharges = np.random.randint(0, 6) if plan_type == 'Prepaid' else 1
                
                # Introduce Usage Drift
                if i % 12 == 0 and m >= months - 2: # Potential churners
                    m_calls = int(m_calls * 0.2)
                    m_data = int(m_data * 0.1)
                    unhappy_factor += 0.15
                
                usage_data.append({
                    'calls': m_calls,
                    'data': m_data,
                    'sms': m_sms,
                    'recharges': m_recharges
                })

            # 5. Service Quality Features
            complaints = np.random.poisson(0.15)
            if complaints > 1: unhappy_factor += 0.2
            
            # 6. Target Variable Calculation (Probabilistic)
            churn_prob = 0.05 + unhappy_factor
            churn = 1 if np.random.random() < min(churn_prob, 0.98) else 0
            
            # 7. Feature Aggregation
            last_m = usage_data[-1]
            prev_m = usage_data[-2]
            
            customers.append({
                'customer_id': cust_id,
                'gender': gender,
                'age': age,
                'state': state,
                'city': city,
                'estimated_salary': salary,
                'plan_type': plan_type,
                'device_type': device,
                'payment_method': payment,
                'tenure_days': tenure_days,
                'total_complaints': complaints,
                'last_month_calls': last_m['calls'],
                'last_month_data': last_m['data'],
                'last_month_sms': last_m['sms'],
                'avg_calls_6m': np.mean([u['calls'] for u in usage_data]),
                'avg_data_6m': np.mean([u['data'] for u in usage_data]),
                'usage_drop_ratio': (last_m['calls'] + 1) / (prev_m['calls'] + 1),
                'recharge_consistency': np.std([u['recharges'] for u in usage_data]),
                'churn': churn
            })

        df = pd.DataFrame(customers)
        
        # Add "Real-world" flaws
        # Missing values (5% of data)
        for col in ['age', 'estimated_salary', 'usage_drop_ratio']:
            df.loc[df.sample(frac=0.05).index, col] = np.nan
            
        # Noise/Outliers
        df.loc[df.sample(frac=0.01).index, 'estimated_salary'] *= 15
        
        logger.info(f"Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    gen = EnhancedDataGenerator()
    data = gen.generate(num_customers=10000)
    data.to_csv('data/raw/telecom_churn_v3.csv', index=False)
    print("Dataset v3 saved.")
