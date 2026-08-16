# 🧠 MindPulse — Mental Health & Lifestyle Analysis

**MindPulse** is an interactive machine-learning web application designed to analyze lifestyle, academic, social, and behavioral factors that may be associated with mental-health-related outcomes.

The application provides two analysis modes:

* 🎓 **Student**
* 👤 **General User**

Users enter information about themselves, and MindPulse uses trained machine-learning models to generate an analysis and provide practical lifestyle suggestions based on the entered factors.

> ⚠️ **Disclaimer:** MindPulse is an educational machine-learning project. It is **not a medical diagnostic tool** and should not be used as a substitute for a qualified doctor, psychologist, psychiatrist, counselor, or other healthcare professional.

---

# 🌐 Live Application

## 🚀 Try MindPulse Online

👉 **[Open the MindPulse Web App](https://mindpulse-et9wcsn3fbyfkqgyvitgek.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud** and can be accessed directly through a web browser.

No local Python installation is required to try the deployed version.

---

# 📌 About MindPulse

MindPulse was developed to explore how machine learning can be applied to lifestyle and behavioral data.

The application focuses on factors such as:

* 🧠 Mental and emotional well-being
* 📚 Academic pressure
* 💼 Work stress
* 😴 Sleep
* 📱 Screen time
* 📲 Social media usage
* 🏃 Physical activity
* 💰 Financial stress
* 🤝 Social support
* 😔 Loneliness
* ⚖️ Work-life balance
* 🎯 Career concerns
* 👥 Social relationships

The goal is to provide users with an easy-to-understand analysis and practical suggestions.

---

# ✨ Main Features

## 🏠 Welcome Page

When the application opens, users first see:

* What MindPulse is
* What the application is designed for
* How the analysis works
* A choice between:

  * 🎓 Student
  * 👤 General User

The application does **not automatically select Student or General User**.

---

# 🎓 Student Analysis

The Student section is designed specifically for students.

It considers academic, social, lifestyle, environmental, and emotional factors.

### Student Inputs

The Student model uses features such as:

* Age
* Self-Esteem
* Mental Health History
* Blood Pressure
* Sleep Quality
* Breathing Problem
* Noise Level
* Living Conditions
* Safety
* Basic Needs
* Academic Performance
* Study Load
* Teacher-Student Relationship
* Future Career Concerns
* Social Support
* Peer Pressure
* Extracurricular Activities
* Bullying

---

# 📊 Mental & Academic Factors

Several Student factors are represented using a **0–10 scale**.

Users can select any value from:

```text
0 1 2 3 4 5 6 7 8 9 10
```

The application does **not automatically select 0**.

Users must choose the value that best represents their situation.

### General Interpretation

| Score | Meaning                |
| ----: | ---------------------- |
|     0 | Very low / Not present |
|   1–2 | Extremely low          |
|   3–4 | Low                    |
|     5 | Moderate               |
|   6–7 | Moderately high        |
|   8–9 | High                   |
|    10 | Extremely high         |

The meaning depends on the individual factor.

---

## 📚 Study Load

Study Load represents how demanding or heavy your academic workload feels.

Example:

* **0** → Almost no study pressure
* **5** → Moderate study workload
* **10** → Extremely heavy study workload

---

## 🎯 Future Career Concerns

Represents how worried or concerned you feel about your future career.

Example:

* **0** → No concern
* **5** → Moderate concern
* **10** → Extremely high concern

---

## 🤝 Social Support

Represents how much support you feel you receive from people around you.

This may include:

* Family
* Friends
* Teachers
* Classmates
* Other trusted people

Example:

* **0** → Almost no support
* **5** → Moderate support
* **10** → Very strong support

---

## 😔 Loneliness

Represents how lonely or socially disconnected you feel.

Example:

* **0** → Not lonely
* **5** → Moderately lonely
* **10** → Extremely lonely

---

## 💪 Self-Esteem

Represents how positively you feel about yourself, your abilities, and your personal worth.

Example:

* **0** → Very low
* **5** → Moderate
* **10** → Very high

---

# 🏠 Living Conditions

**Living Conditions** refers to the quality and comfort of the environment where you currently live.

It may include factors such as:

* Comfort
* Cleanliness
* Stability
* Privacy
* Availability of basic facilities
* Overall suitability of the living environment

A higher or lower score should reflect how you personally experience your living environment.

---

# 🛌 Sleep Inputs

Sleep-related inputs are **not treated as general 0–10 rankings when the model expects an actual quantity**.

For example:

### Sleep Hours

Users provide the approximate number of hours they sleep per night.

Examples:

```text
4 hours
6 hours
7 hours
8 hours
```

This represents an actual duration.

---

# 🧠 Mental Health History

Mental Health History is a **Yes / No** question.

It is **not a 0–10 rating**.

Users can select:

* Yes
* No

This represents whether the user reports a previous mental-health history.

---

# 👤 Gender

Gender is treated as a categorical input.

The application does **not automatically select Male**.

Users choose the appropriate option themselves.

---

# 👤 General User Analysis

The General User section is designed for people who are not specifically using the Student analysis.

It focuses on lifestyle, work, financial, social, and behavioral factors.

### General User Inputs

The model uses features such as:

* Age
* Gender
* Income Level
* Employment Status
* Work Hours Per Week
* Job Satisfaction
* Work Stress Level
* Work-Life Balance
* Exercise Per Week
* Sleep Hours Per Night
* Screen Time Per Day
* Social Media Hours Per Day
* Hobby Time Per Week
* Financial Stress
* Social Support
* Close Friends Count
* Feel Understood
* Loneliness

---

# 📱 Screen Time

Screen Time represents the approximate number of hours spent using digital screens each day.

This can include:

* Smartphone
* Laptop
* Computer
* Tablet
* Other digital devices

It is represented as an actual number of hours rather than a simple 0–10 rating.

---

# 📲 Social Media Usage

Social Media Hours represents the approximate number of hours spent on social media each day.

Examples include:

* Instagram
* YouTube
* Facebook
* X
* Snapchat
* Other social platforms

---

# 🏃 Exercise

Exercise represents the amount/frequency of physical activity.

Examples may include:

* Walking
* Running
* Gym
* Cycling
* Sports
* Outdoor games

Regular physical activity can be a useful part of maintaining a healthy lifestyle.

---

# 💡 Personalized Suggestions

After analyzing the user's responses, MindPulse provides suggestions based on the factors that may require attention.

The suggestions are intended as **general lifestyle guidance**, not medical treatment.

---

## 📱 High Phone or Social Media Usage

If excessive screen or social-media usage is reported, MindPulse may suggest:

* Reduce unnecessary phone usage.
* Limit excessive social media scrolling.
* Take regular screen breaks.
* Avoid using the phone continuously for long periods.
* Spend more time on offline activities.
* Try hobbies, sports, or outdoor activities.

---

## 🏃 Low Physical Activity

MindPulse may suggest:

* Try regular exercise.
* Go for a walk.
* Play outdoor games.
* Try cycling, running, or sports.
* Include more physical movement in your daily routine.

---

## 😴 Poor Sleep

Suggestions may include:

* Maintain a consistent sleep schedule.
* Try to get sufficient sleep.
* Reduce screen usage before sleeping.
* Avoid unnecessary late-night activities.
* Create a comfortable sleeping environment.

---

## 📚 High Study Load

Suggestions may include:

* Break large tasks into smaller tasks.
* Create a realistic study timetable.
* Take short breaks.
* Avoid studying continuously for very long periods.
* Prioritize important academic tasks.

---

## 💼 High Work Stress

Suggestions may include:

* Take regular breaks.
* Maintain a healthier work-life balance.
* Avoid unnecessary overtime when possible.
* Make time for hobbies.
* Include physical activity in your routine.
* Spend time away from work-related screens.

---

## 💰 High Financial Stress

Suggestions may include:

* Create a simple monthly budget.
* Track essential and non-essential spending.
* Identify unnecessary expenses.
* Set realistic financial priorities.
* Seek appropriate financial guidance when necessary.

---

## 😔 High Loneliness

Suggestions may include:

* Talk with trusted friends or family.
* Participate in group activities.
* Play sports or outdoor games.
* Join clubs or communities.
* Spend more time with supportive people.

---

## 🤝 Low Social Support

Suggestions may include:

* Talk with someone you trust.
* Stay connected with friends and family.
* Participate in social activities.
* Build healthy relationships.
* Seek professional support if you feel overwhelmed.

---

# 🤖 Machine Learning

MindPulse was developed using Python and several machine-learning techniques.

The project involved:

* Data cleaning
* Exploratory Data Analysis
* Feature analysis
* Feature selection
* Model training
* Model comparison
* Class balancing
* SMOTE experimentation
* Hyperparameter tuning
* Threshold analysis
* Cross-validation
* Model evaluation
* Streamlit deployment

---

# 🎓 Student Machine Learning Model

The Student model was trained using student stress-related data.

Important features include:

```text
self_esteem
mental_health_history
blood_pressure
sleep_quality
breathing_problem
noise_level
living_conditions
safety
basic_needs
academic_performance
study_load
teacher_student_relationship
future_career_concerns
social_support
peer_pressure
extracurricular_activities
bullying
```

The Student model is integrated into the Streamlit application.

---

# 👤 General User Machine Learning Model

The General User model was developed through multiple stages.

## Model Development Process

```text
Initial General User Model
        ↓
Feature Analysis
        ↓
Balanced Model
        ↓
Selected Feature Model
        ↓
SMOTE Experiment
        ↓
Model Evaluation
        ↓
Threshold Analysis
        ↓
Improved Model
        ↓
Hyperparameter Tuning
        ↓
Tuned Model
        ↓
Tuned Threshold Analysis
        ↓
Final Application
```

---

# 📈 General User Model Results

The dataset used for the General User model contained:

```text
Dataset Shape: 10,000 rows × 51 columns
```

The target distribution was highly imbalanced:

```text
Mental Health Issue = 1 → 92.16%
Mental Health Issue = 0 → 7.84%
```

Because of this imbalance, simply using Accuracy is not sufficient.

Therefore, the project also considered:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Balanced Accuracy
* Confusion Matrix

---

# 🔧 Hyperparameter Tuned Model

The final tuned General User model selected a Random Forest approach based on Balanced Accuracy.

### Tuned Random Forest

```text
Model:
Random Forest

Class Weight:
balanced

Max Depth:
5

Minimum Samples Leaf:
1

Number of Estimators:
200
```

### Test Performance

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 66.35% |
| Precision         | 94.05% |
| Recall            | 67.77% |
| F1 Score          | 78.78% |
| ROC-AUC           | 61.03% |
| Balanced Accuracy | 58.73% |

---

# ⚖️ Why Balanced Accuracy?

The General User dataset is highly imbalanced.

A model could achieve high accuracy simply by predicting the majority class for most users.

For example, a model predicting almost everyone as having a mental-health issue can obtain around 92% accuracy in this dataset.

However, that does not mean the model is performing well.

Therefore, MindPulse considers **Balanced Accuracy** along with:

* Recall
* Precision
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

---

# 📊 Threshold Analysis

The tuned General User model was also evaluated using different prediction thresholds.

The selected threshold was:

```text
0.50
```

At this threshold:

```text
Accuracy          : 66.35%
Precision         : 94.05%
Recall            : 67.77%
F1 Score          : 78.78%
Balanced Accuracy : 58.73%
ROC-AUC           : 61.03%
```

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
│   └── processed/
│
├── models/
│   │
│   ├── student_features.pkl
│   ├── student_stress_model.pkl
│   │
│   ├── general_user_model.pkl
│   ├── general_user_features.pkl
│   │
│   ├── general_user_balanced_features.pkl
│   ├── general_user_balanced_model.pkl
│   │
│   ├── general_user_selected_features.pkl
│   ├── general_user_selected_model.pkl
│   │
│   ├── general_user_smote_features.pkl
│   ├── general_user_smote_model.pkl
│   │
│   ├── general_user_improved_features.pkl
│   ├── general_user_improved_model.pkl
│   │
│   ├── general_user_tuned_features.pkl
│   ├── general_user_tuned_model.pkl
│   ├── general_user_tuning_results.csv
│   │
│   ├── general_user_tuned_threshold_results.csv
│   └── general_user_tuned_threshold.pkl
│
├── src/
│   │
│   ├── analyze_general_features.py
│   ├── analyze_general_target.py
│   ├── inspect_global_dataset.py
│   │
│   ├── train_general_balanced.py
│   ├── train_general_selected_model.py
│   ├── train_general_smote_model.py
│   │
│   ├── evaluate_general_model.py
│   ├── analyze_general_threshold.py
│   ├── improve_general_model.py
│   ├── tune_general_model.py
│   └── analyze_tuned_threshold.py
│
└── reports/
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

### Development Tools

* VS Code
* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

# 🚀 Running MindPulse Locally

## 1. Clone the repository

```bash
git clone https://github.com/ritura17/Mindpulse.git
```

## 2. Open the project

```bash
cd Mindpulse
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🌐 Online Usage

You can use the deployed application here:

👉 **https://mindpulse-et9wcsn3fbyfkqgyvitgek.streamlit.app/**

Recommended flow:

```text
Open MindPulse
      ↓
Read About MindPulse
      ↓
Select Student / General User
      ↓
Enter personal information
      ↓
Enter lifestyle / academic / work factors
      ↓
Click Analyze
      ↓
Receive prediction
      ↓
View risk/result information
      ↓
Read personalized suggestions
```

---

# 🔬 Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Feature Analysis
      ↓
Model Training
      ↓
Model Comparison
      ↓
Class Imbalance Handling
      ↓
Hyperparameter Tuning
      ↓
Threshold Analysis
      ↓
Model Evaluation
      ↓
Streamlit Application
      ↓
Deployment
```

---

# 📊 Model Evaluation

The project evaluates models using multiple metrics.

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures how well the model separates the two classes across different thresholds.

### PR-AUC

Useful when dealing with imbalanced classification datasets.

### Balanced Accuracy

Calculates performance across both classes and is particularly useful for imbalanced datasets.

---

# ⚠️ Limitations

MindPulse has several important limitations:

* The dataset may not perfectly represent every population.
* The General User target is highly imbalanced.
* Model performance depends on the quality of the training data.
* Predictions are statistical estimates, not medical diagnoses.
* Self-reported information can contain bias.
* The application cannot understand every personal or psychological situation.
* A prediction should not be interpreted as a clinical assessment.

---

# 🔐 Privacy

MindPulse is designed as an educational application.

Users should avoid entering highly sensitive personal information that is not required by the application.

Do not enter:

* Passwords
* Financial account information
* Government identification numbers
* Private medical records
* Other unnecessary sensitive information

---

# 🎯 Future Improvements

Possible future improvements include:

* Improve dataset quality and class balance
* Collect more diverse data
* Improve minority-class detection
* Experiment with advanced ML models
* Improve probability calibration
* Add explainable AI
* Add feature importance visualization
* Add SHAP explanations
* Improve personalized recommendations
* Add historical analysis
* Add downloadable reports
* Improve mobile UI
* Add multilingual support
* Add more student-specific analysis
* Improve model validation using external datasets

---

# 👨‍💻 Author

**Rituraj Kumar**

B.Tech Computer Engineering Student

GitHub:

👉 **[github.com/ritura17](https://github.com/ritura17)**

---

# ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

### 🔗 Links

* 🌐 **[Live MindPulse App](https://mindpulse-et9wcsn3fbyfkqgyvitgek.streamlit.app/)**
* 💻 **[GitHub Repository](https://github.com/ritura17/Mindpulse)**

---

# ⚠️ Final Disclaimer

MindPulse is intended for **educational, research, and demonstration purposes only**.

The predictions and suggestions provided by the application should not be considered medical, psychological, or clinical advice.

If someone is experiencing serious or persistent mental-health difficulties, they should consider speaking with a qualified healthcare or mental-health professional.
