import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return plt

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return plt

def plot_lift_chart(y_true, y_prob):
    """
    Generates a Lift Chart to show how many times better the model is than random.
    """
    df = pd.DataFrame({'actual': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False)
    df['cum_actual'] = df['actual'].cumsum() / df['actual'].sum()
    df['cum_random'] = np.linspace(0, 1, len(df))
    
    # Lift = Actual Cumulative / Random Cumulative
    # Usually visualized in deciles
    df['decile'] = pd.qcut(df['prob'].rank(method='first'), 10, labels=False)
    decile_stats = df.groupby('decile')['actual'].agg(['count', 'sum']).sort_index(ascending=False)
    decile_stats['cum_sum'] = decile_stats['sum'].cumsum()
    decile_stats['lift'] = (decile_stats['cum_sum'] / decile_stats['sum'].sum()) / \
                           (decile_stats['count'].cumsum() / len(df))
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), decile_stats['lift'], color='teal')
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.title('Model Lift Chart')
    return plt

def plot_profit_curve(y_true, y_prob, cost_miss=500, cost_fp=50, profit_tp=400):
    """
    Visualizes profit across different decision thresholds.
    """
    thresholds = np.linspace(0, 1, 100)
    profits = []
    
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = (tp * profit_tp) - (fn * cost_miss) - (fp * cost_fp)
        profits.append(profit)
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, lw=3, color='green')
    plt.axvline(thresholds[np.argmax(profits)], color='red', linestyle='--', label=f'Max Profit at t={thresholds[np.argmax(profits)]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Projected Profit ($)')
    plt.title('Business Profit Curve')
    plt.legend()
    return plt
