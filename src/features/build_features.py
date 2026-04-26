import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for domain-specific feature engineering."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Tenure Buckets
        X['tenure_group'] = pd.cut(X['tenure_months'], 
                                   bins=[0, 6, 12, 24, 1000], 
                                   labels=['New', 'Junior', 'Senior', 'Legacy'])
        
        # 2. Interaction Features
        X['data_per_call'] = X['last_month_data'] / (X['last_month_calls'] + 1)
        X['salary_usage_index'] = X['estimated_salary'] / (X['last_month_data'] + X['last_month_calls'] + 1)
        
        # 3. High Value Customer Tag
        X['is_high_value'] = (X['estimated_salary'] > X['estimated_salary'].median()) & \
                             (X['last_month_data'] > X['last_month_data'].median())
        X['is_high_value'] = X['is_high_value'].astype(int)
        
        return X

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """Creates a full preprocessing pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Combined pipeline with custom engineering
    full_pipeline = Pipeline(steps=[
        ('engineer', FeatureEngineer()),
        ('preprocess', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    # Test loading
    df = pd.read_csv('data/raw/telecom_churn_v2.csv')
    num_cols = ['age', 'estimated_salary', 'tenure_months', 'total_complaints', 
                'avg_calls_3m', 'avg_data_3m', 'usage_drop_ratio', 'recharge_frequency']
    cat_cols = ['gender', 'state', 'city', 'plan_type', 'tenure_group'] # tenure_group added by engineer
    
    # Note: tenure_group is added during transformation, so we need to be careful with ColumnTransformer
    # Actually, it's better to define columns after the engineer step.
    pass
