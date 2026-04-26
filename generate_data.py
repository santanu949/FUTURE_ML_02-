import pandas as pd
import numpy as np

def generate_dummy_data(num_records=1000):
    partners = ['Reliance Jio', 'Vodafone', 'BSNL', 'Airtel']
    states = ['Karnataka', 'Mizoram', 'Arunachal Pradesh', 'Tamil Nadu', 'Tripura', 'West Bengal', 'Maharashtra', 'Delhi']
    cities = ['Kolkata', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore']
    
    data = {
        'customer_id': range(1, num_records + 1),
        'telecom_partner': np.random.choice(partners, num_records),
        'gender': np.random.choice(['M', 'F'], num_records),
        'age': np.random.randint(18, 80, num_records),
        'state': np.random.choice(states, num_records),
        'city': np.random.choice(cities, num_records),
        'pincode': np.random.randint(100000, 999999, num_records),
        'date_of_registration': ['2020-01-01'] * num_records,
        'num_dependents': np.random.randint(0, 5, num_records),
        'estimated_salary': np.random.randint(20000, 200000, num_records),
        'calls_made': np.random.randint(0, 200, num_records),
        'sms_sent': np.random.randint(0, 100, num_records),
        'data_used': np.random.randint(-500, 10000, num_records),
        'churn': np.random.choice([0, 1], num_records, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('telecom_churn.csv', index=False)
    print("Generated telecom_churn.csv with {} records.".format(num_records))

if __name__ == "__main__":
    generate_dummy_data()
