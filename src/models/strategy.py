import pandas as pd
import numpy as np

class RetentionStrategy:
    def __init__(self, churn_model):
        self.model = churn_model
        self.intervention_costs = {
            'High Discount': 100,
            'Support Call': 20,
            'Standard Offer': 50
        }
        self.success_rates = {
            'High Discount': 0.6,
            'Support Call': 0.3,
            'Standard Offer': 0.4
        }

    def simulate_roi(self, X_test, y_true):
        # 1. Predict Churn Probability
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = self.model.predict(X_test)
        
        results = pd.DataFrame({
            'actual_churn': y_true,
            'churn_prob': probs,
            'predicted_churn': preds
        })
        
        # 2. Assign Strategy based on Risk
        def assign_strategy(prob):
            if prob > 0.8: return 'High Discount'
            if prob > 0.5: return 'Standard Offer'
            if prob > 0.3: return 'Support Call'
            return 'None'
        
        results['strategy'] = results['churn_prob'].apply(assign_strategy)
        
        # 3. Calculate ROI
        # Revenue saved = Success Rate * Customer Lifetime Value (assume $1000)
        # Net Profit = Revenue Saved - Intervention Cost
        
        clv = 1000
        
        def calculate_net_gain(row):
            if row['strategy'] == 'None': return 0
            
            cost = self.intervention_costs[row['strategy']]
            success_rate = self.success_rates[row['strategy']]
            
            if row['actual_churn'] == 1:
                return (success_rate * clv) - cost
            else:
                return -cost # Lost marketing money on non-churner
                
        results['net_gain'] = results.apply(calculate_net_gain, axis=1)
        
        total_roi = results['net_gain'].sum()
        print(f"\n--- Retention ROI Simulation ---")
        print(f"Total Customers Targeted: {len(results[results['strategy'] != 'None'])}")
        print(f"Projected Net Profit from Retention: ${total_roi:,.2f}")
        
        return results

if __name__ == "__main__":
    import joblib
    model = joblib.load('models/churn_model_v2.pkl')
    # Use dummy X_test for now
    pass
