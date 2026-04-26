# Telecom Customer Churn Prediction 📡📱

## 📌 Project Overview
This project focuses on predicting **Customer Churn** in the telecommunications industry. Using a dataset of over 240,000 customer records, we apply machine learning techniques to identify individuals who are likely to discontinue their service. Predicting churn allows companies to take proactive measures, such as offering personalized discounts or improving service quality, to retain high-value customers.

## 🚀 Why This Project?
- **Business Impact**: Retaining existing customers is significantly cheaper than acquiring new ones.
- **Data-Driven Insights**: Understand the key factors (age, usage patterns, salary, etc.) that influence customer loyalty.
- **Machine Learning Mastery**: Demonstrates handling large-scale datasets, addressing class imbalance (SMOTE), and comparing multiple classification models.

---

## 🛠️ Tech Stack & Dependencies
The project is built using **Python** and the following libraries:
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Imbalance Handling**: `imbalanced-learn` (SMOTE)

---

## 📂 Project Structure
```text
FUTURE_ML_02--main/
├── FUTURE_ML_02_.ipynb   # Main Jupyter Notebook with code and analysis
├── generate_data.py      # Synthetic data generator
├── verify_project.py     # Project verification script
├── telecom_churn.csv     # Sample dataset (generated)
└── README.md             # Project documentation (this file)
```

### 🧩 Major Components
1. **Data Loading & Exploration**: Initial inspection of the dataset schema, summary statistics, and visualization of churn distribution.
2. **Preprocessing**: 
   - **Encoding**: Converting categorical variables (gender, telecom partner, state, city) into numerical formats using mapping and One-Hot Encoding.
   - **Feature Selection**: Dropping non-predictive columns like `customer_id` and `date_of_registration`.
3. **Handling Imbalance**: Using **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the churned vs. non-churned classes.
4. **Modeling**:
   - **Logistic Regression**: A baseline classification model.
   - **Random Forest**: An ensemble method for capturing non-linear relationships.
5. **Evaluation**: Assessing performance using Precision, Recall, F1-Score, and ROC-AUC metrics.
6. **Utility Scripts**:
   - `generate_data.py`: A utility script to create `telecom_churn.csv` if the original data is missing.
   - `verify_project.py`: A standalone Python script to verify the ML pipeline without needing Jupyter.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You can download it from [python.org](https://www.python.org/).

### 2. Clone the Repository
```bash
git clone https://github.com/santanu949/FUTURE_ML_02-.git
cd FUTURE_ML_02--main
```

### 3. Install Dependencies
```bash
pip install pandas matplotlib scikit-learn imbalanced-learn
```

### 4. Prepare the Data
The notebook expects a file named `telecom_churn.csv` in the same directory. 
> **Note**: If you don't have the dataset, ensure you provide a CSV with columns: `gender`, `age`, `telecom_partner`, `state`, `city`, `estimated_salary`, `calls_made`, `sms_sent`, `data_used`, and `churn`.

### 5. Run the Project
Open the Jupyter Notebook:
```bash
jupyter notebook FUTURE_ML_02--main/FUTURE_ML_02_.ipynb
```
Or use Google Colab to run the `.ipynb` file directly.

---

## 🔄 Workflow & Data Flow
1. **Input**: Raw telecom data (`.csv`).
2. **Analysis**: Exploratory Data Analysis (EDA) to find trends.
3. **Transformation**: Mapping binary features and dummifying multi-class features.
4. **Oversampling**: Generating synthetic samples for the minority class (Churn=1).
5. **Training**: Training models on the resampled training set.
6. **Prediction**: Running the model on a hidden test set.
7. **Output**: Performance reports and confusion matrices.

---

## 📈 Results
- **Logistic Regression**: Provided a baseline accuracy but struggled with the minority class before balancing.
- **SMOTE Impact**: Improved the recall for churned customers significantly.
- **Random Forest**: Demonstrated better handling of the complex feature set with competitive ROC-AUC scores.

---

## 🤝 Contributing
Feel free to fork this project, open issues, or submit pull requests to improve the model's accuracy!

---
**Author**: Santanu
**Internship**: Future Interns (Machine Learning Task 2)
