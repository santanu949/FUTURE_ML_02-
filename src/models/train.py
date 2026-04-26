import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import joblib
from sklearn.pipeline import Pipeline
from src.features.build_features import get_preprocessing_pipeline

def business_cost_score(y_true, y_pred):
    """
    Penalize False Negatives (missing a churner) more than False Positives.
    Cost of missing a churner = $500 (lost revenue)
    Cost of false alarm = $50 (marketing cost)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * 500) + (fp * 50)
    return total_cost

def train_production_model():
    # 1. Load Data
    data_path = 'data/raw/telecom_churn_v2.csv'
    if not os.path.exists(data_path):
        from src.data.generator import TelecomDataGenerator
        gen = TelecomDataGenerator()
        df = gen.generate(num_customers=5000)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Setup Pipeline
    num_cols = ['age', 'estimated_salary', 'tenure_months', 'total_complaints', 
                'avg_calls_3m', 'avg_data_3m', 'usage_drop_ratio', 'recharge_frequency']
    cat_cols = ['gender', 'state', 'city', 'plan_type', 'tenure_group']

    preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)

    # 4. Model & Hyperparams
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0]
    }

    print("--- Starting Hyperparameter Tuning ---")
    search = RandomizedSearchCV(full_pipeline, param_dist, n_iter=5, cv=3, scoring='roc_auc', verbose=1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    
    # 5. Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Business Cost: ${business_cost_score(y_test, y_pred)}")

    # 6. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model_v2.pkl')
    print("\nModel saved to models/churn_model_v2.pkl")

    return best_model

if __name__ == "__main__":
    train_production_model()
