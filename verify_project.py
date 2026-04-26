import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

def run_pipeline():
    file_path = 'telecom_churn.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run generate_data.py first.")
        return

    print("--- Loading Data ---")
    data = pd.read_csv(file_path)
    print("Dataset shape:", data.shape)

    print("\n--- Preprocessing ---")
    # Mapping gender
    data['gender'] = data['gender'].map({'M': 0, 'F': 1})
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['telecom_partner', 'state', 'city'], drop_first=True)
    
    # Feature selection
    X = data.drop(['customer_id', 'date_of_registration', 'churn'], axis=1)
    y = data['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    print("\n--- Applying SMOTE ---")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE, counts of label '1':", sum(y_train_res == 1))
    print("After SMOTE, counts of label '0':", sum(y_train_res == 0))

    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_res, y_train_res)
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    print("ROC AUC Score (LR):", roc_auc_score(y_test, y_proba_lr))

    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_res, y_train_res)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    print("ROC AUC Score (RF):", roc_auc_score(y_test, y_proba_rf))
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    run_pipeline()
