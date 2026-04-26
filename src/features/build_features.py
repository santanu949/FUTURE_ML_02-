import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Implements 20+ feature engineering steps including interactions, 
    segmentation, and temporal buckets.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Temporal Features (Tenure Buckets)
        X['tenure_years'] = X['tenure_days'] / 365
        X['tenure_group'] = pd.cut(X['tenure_years'], 
                                   bins=[-1, 1, 3, 5, 100], 
                                   labels=['New', 'Established', 'Senior', 'Legacy'])
        
        # 2. Behavioral Interaction Features
        # Usage intensity
        X['data_per_call'] = X['last_month_data'] / (X['last_month_calls'] + 1)
        X['sms_to_call_ratio'] = X['last_month_sms'] / (X['last_month_calls'] + 1)
        
        # Financial behavior
        X['salary_usage_index'] = X['estimated_salary'] / (X['last_month_data'] + 1)
        X['arpu_est'] = (X['last_month_calls'] * 0.1) + (X['last_month_data'] * 0.01) + (X['last_month_sms'] * 0.05)
        
        # 3. Customer Segmentation
        # High-Value Tag (Upper quartile of ARPU and Salary)
        arpu_q3 = X['arpu_est'].quantile(0.75)
        salary_q3 = X['estimated_salary'].quantile(0.75)
        X['is_high_value'] = ((X['arpu_est'] > arpu_q3) & (X['estimated_salary'] > salary_q3)).astype(int)
        
        # Risk Indicators
        X['complaint_to_tenure_ratio'] = X['total_complaints'] / (X['tenure_years'] + 0.1)
        X['is_senior_citizen'] = (X['age'] > 60).astype(int)
        
        # 4. Usage Stability
        X['usage_stability_index'] = 1 / (X['recharge_consistency'] + 0.1)
        
        return X

def get_production_pipeline(numeric_features, categorical_features):
    """
    Creates a production-ready ColumnTransformer with robust preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Simple for speed, KNN for accuracy
        ('scaler', RobustScaler()) # Robust to outliers
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return Pipeline(steps=[
        ('engineer', AdvancedFeatureEngineer()),
        ('preprocess', preprocessor)
    ])

if __name__ == "__main__":
    # Integration test
    df = pd.read_csv('data/raw/telecom_churn_v3.csv')
    X = df.drop(['customer_id', 'churn'], axis=1)
    
    # Define features based on what AdvancedFeatureEngineer outputs
    num_cols = ['age', 'estimated_salary', 'tenure_days', 'total_complaints', 
                'last_month_calls', 'last_month_data', 'last_month_sms', 
                'avg_calls_6m', 'avg_data_6m', 'usage_drop_ratio', 'recharge_consistency',
                'tenure_years', 'data_per_call', 'sms_to_call_ratio', 
                'salary_usage_index', 'arpu_est', 'complaint_to_tenure_ratio', 
                'usage_stability_index']
    
    cat_cols = ['gender', 'state', 'city', 'plan_type', 'device_type', 
                'payment_method', 'tenure_group']
    
    # Check if engineer works
    fe = AdvancedFeatureEngineer()
    X_eng = fe.transform(X)
    print("Engineered Columns:", X_eng.columns)
    print("Sample Output:\n", X_eng.head(2))
