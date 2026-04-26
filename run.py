import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Telecom Churn Production Pipeline CLI")
    parser.add_argument("command", choices=["generate", "train", "api", "app"], help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        print("Generating data...")
        from src.data.generator import TelecomDataGenerator
        gen = TelecomDataGenerator()
        df = gen.generate(num_customers=5000)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/telecom_churn_v2.csv', index=False)
        print("Done.")
        
    elif args.command == "train":
        print("Training model...")
        from src.models.train import train_production_model
        train_production_model()
        print("Done.")
        
    elif args.command == "api":
        print("Starting FastAPI server...")
        os.system("uvicorn src.api.main:app --reload")
        
    elif args.command == "app":
        print("Starting Streamlit dashboard...")
        os.system("streamlit run src.app.dashboard.py")

if __name__ == "__main__":
    main()
