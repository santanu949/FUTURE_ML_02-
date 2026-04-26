import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer

def test_feature_engineer_tenure_buckets():
    fe = FeatureEngineer()
    df = pd.DataFrame({
        'tenure_months': [1, 7, 13, 30],
        'last_month_data': [100, 200, 300, 400],
        'last_month_calls': [10, 20, 30, 40],
        'estimated_salary': [50000, 60000, 70000, 80000]
    })
    
    transformed = fe.transform(df)
    
    assert 'tenure_group' in transformed.columns
    assert list(transformed['tenure_group']) == ['New', 'Junior', 'Senior', 'Legacy']

def test_feature_engineer_interactions():
    fe = FeatureEngineer()
    df = pd.DataFrame({
        'tenure_months': [12],
        'last_month_data': [1000],
        'last_month_calls': [9],
        'estimated_salary': [10000]
    })
    
    transformed = fe.transform(df)
    
    # data_per_call = 1000 / (9 + 1) = 100
    assert transformed['data_per_call'].iloc[0] == 100.0

if __name__ == "__main__":
    pytest.main([__file__])
