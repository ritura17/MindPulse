# 🧠 MindPulse - Mental Health & Behavioral Risk Analysis

MindPulse is a machine learning project designed to analyze behavioral, lifestyle, social, and work-related factors associated with mental health risk.

The project contains two major prediction modules:

- 🎓 Student Stress Analysis
- 👤 General User Mental Health Risk Analysis

The project uses Python, Pandas, NumPy, Scikit-learn, Imbalanced-Learn, Joblib, and Streamlit.

---

## 📌 Project Overview

Mental health can be influenced by several behavioral and lifestyle factors such as:

- Work stress
- Financial stress
- Social support
- Loneliness
- Sleep duration
- Screen time
- Social media usage
- Exercise
- Work-life balance
- Job satisfaction
- Social interaction

MindPulse uses machine learning models to identify patterns in these factors and estimate mental health or stress risk.

> ⚠️ MindPulse is an educational machine learning project and is not a medical diagnostic system.

---

# 🎯 Project Objectives

The main objectives of MindPulse are:

1. Analyze mental health and lifestyle datasets.
2. Identify important behavioral risk factors.
3. Build machine learning models for student stress prediction.
4. Build machine learning models for general-user mental health risk prediction.
5. Handle class imbalance.
6. Compare multiple machine learning algorithms.
7. Perform hyperparameter tuning.
8. Perform threshold analysis.
9. Evaluate models using appropriate classification metrics.
10. Deploy the final model through a Streamlit application.

---

# 📂 Project Structure

```text
Mindpulse/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── Global_Mental_Health_Lifestyle_Survey.csv
│   │   └── student_stress.csv
│   │
│   └── processed/
│
├── models/
│   ├── general_user_model.pkl
│   ├── general_user_features.pkl
│   │
│   ├── general_user_balanced_model.pkl
│   ├── general_user_balanced_features.pkl
│   │
│   ├── general_user_selected_model.pkl
│   ├── general_user_selected_features.pkl
│   │
│   ├── general_user_smote_model.pkl
│   ├── general_user_smote_features.pkl
│   │
│   ├── general_user_improved_model.pkl
│   ├── general_user_improved_features.pkl
│   │
│   ├── general_user_tuned_model.pkl
│   ├── general_user_tuned_features.pkl
│   ├── general_user_tuning_results.csv
│   ├── general_user_tuned_threshold_results.csv
│   └── general_user_tuned_threshold.pkl
│
├── src/
│   ├── inspect_global_dataset.py
│   ├── analyze_general_target.py
│   ├── analyze_general_features.py
│   │
│   ├── train_general_model.py
│   ├── train_general_balanced.py
│   ├── train_general_selected_model.py
│   ├── train_general_smote_model.py
│   ├── train_general_improved_model.py
│   │
│   ├── evaluate_general_model.py
│   ├── analyze_general_threshold.py
│   ├── tune_general_model.py
│   └── analyze_general_tuned_threshold.py
│
└── reports/