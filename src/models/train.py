import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import os
import joblib
import logging
from sklearn.pipeline import Pipeline
from src.features.build_features import get_production_pipeline

logger = logging.getLogger(__name__)

def evaluate_business_impact(y_true, y_prob, threshold=0.5):
    """
    Calculates the financial impact of the model.
    """
    y_pred = (y_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Costs
    cost_miss = fn * 450 # Revenue lost from churner we missed
    cost_marketing = fp * 40 # Cost of offering discount to loyalist
    saving = tp * 300 # Net saving from retained churner (Revenue - Discount)
    
    net_impact = saving - cost_miss - cost_marketing
    return net_impact

def train_ensemble_pipeline():
    # 1. Load Data
    data_path = 'data/raw/telecom_churn_v3.csv'
    if not os.path.exists(data_path):
        from src.data.generator import EnhancedDataGenerator
        gen = EnhancedDataGenerator()
        df = gen.generate(num_customers=5000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Setup Preprocessing
    # Define columns for ColumnTransformer (based on Engineer output)
    num_cols = ['age', 'estimated_salary', 'tenure_days', 'total_complaints', 
                'last_month_calls', 'last_month_data', 'last_month_sms', 
                'avg_calls_6m', 'avg_data_6m', 'usage_drop_ratio', 'recharge_consistency',
                'tenure_years', 'data_per_call', 'sms_to_call_ratio', 
                'salary_usage_index', 'arpu_est', 'complaint_to_tenure_ratio', 
                'usage_stability_index']
    
    cat_cols = ['gender', 'state', 'city', 'plan_type', 'device_type', 
                'payment_method', 'tenure_group']

    preprocessor = get_production_pipeline(num_cols, cat_cols)

    # 4. Define Base Models for Stacking
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbosity=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ]

    # Stacking Classifier with Logistic Regression as meta-learner
    stack_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacking', stack_clf)
    ])

    # 5. Training with Cross-Validation
    logger.info("Starting Ensemble Stacking Training with 5-fold CV...")
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    full_pipeline.fit(X_train, y_train)

    # 6. Detailed Evaluation
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    y_pred = full_pipeline.predict(X_test)

    logger.info("\n--- Production Model Performance ---")
    logger.info(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    impact = evaluate_business_impact(y_test, y_prob)
    logger.info(f"Estimated Business Impact (Net Saving): ${impact:,.2f}")

    # 7. Persistence
    os.makedirs('models', exist_ok=True)
    joblib.dump(full_pipeline, 'models/churn_ensemble_v3.pkl')
    logger.info("Model v3 saved successfully.")

    return full_pipeline

if __name__ == "__main__":
    train_ensemble_pipeline()
