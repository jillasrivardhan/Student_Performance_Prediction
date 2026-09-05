# 🎓 Student Performance Prediction

A machine learning classification project that predicts a student's **final academic grade** using academic, behavioral, and lifestyle-related factors.

The project demonstrates an end-to-end machine learning workflow using **Python, Pandas, Scikit-learn, and Streamlit**, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, cross-validation, model comparison, and deployment through an interactive web application.

---

## 🚀 Project Overview

Student academic performance can be influenced by several factors such as:

* Study time
* Attendance
* Previous academic performance
* Final examination score
* Sleep duration
* Parental education
* Internet access
* Extracurricular activities
* Part-time employment
* Gender

This project uses these features to build a machine learning classification system capable of predicting a student's final grade category.

### 🎯 Prediction Classes

| Encoded Class | Grade |
| ------------: | :---: |
|           `0` |   A   |
|           `1` |   B   |
|           `2` |   C   |
|           `3` |   D   |
|           `4` |   F   |

---

## ✨ Features

* 📊 Exploratory Data Analysis
* 🧹 Data preprocessing and cleaning
* 🔢 Binary categorical encoding
* 🏷️ Target label encoding
* 🔄 Automatic numerical/categorical feature detection
* 🛠️ Missing-value imputation
* 📏 Numerical feature scaling
* 🔠 One-hot encoding
* 🔗 Scikit-learn preprocessing pipeline
* 🤖 Multiple classification algorithms
* 📈 Accuracy evaluation
* 📋 Classification report
* 🔲 Confusion matrix
* 🔁 5-fold cross-validation
* 🌐 Interactive Streamlit application
* 💾 Saved trained machine-learning model

---

# 🧠 Machine Learning Workflow

```text
                    Student Dataset
                          │
                          ▼
                   Data Exploration
                          │
                          ▼
                    Data Cleaning
                          │
                          ▼
              Feature / Target Separation
                          │
                          ▼
                 Feature Preprocessing
                    ┌─────┴─────┐
                    ▼           ▼
               Numerical    Categorical
                    │           │
              Imputation     Imputation
                    │           │
                 Scaling    One-Hot Encoding
                    └─────┬─────┘
                          ▼
                 Train-Test Split
                          │
                          ▼
                 Model Training
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Logistic     Random      Gradient
         Regression     Forest      Boosting
             │
             ├── SVM
             ├── KNN
             ├── Naive Bayes
             └── XGBoost
                          │
                          ▼
                  Model Evaluation
                          │
                          ▼
                 Trained Model
                          │
                          ▼
                  Streamlit Web App
                          │
                          ▼
                 Final Grade Prediction
```

---

# 📊 Dataset

The project uses:

```text
student_performance_dataset.csv
```

The dataset contains **1,000 student records** and **12 columns**.

### Dataset Features

| Feature                      | Description                                 | Type        |
| ---------------------------- | ------------------------------------------- | ----------- |
| `student_id`                 | Unique student identifier                   | Numerical   |
| `gender`                     | Student gender                              | Categorical |
| `study_time_hours`           | Daily study time                            | Numerical   |
| `attendance_percent`         | Attendance percentage                       | Numerical   |
| `sleep_hours`                | Daily sleep duration                        | Numerical   |
| `parental_education`         | Parent's education level                    | Categorical |
| `internet_access`            | Internet availability                       | Binary      |
| `extracurricular_activities` | Participation in extracurricular activities | Binary      |
| `part_time_job`              | Whether the student has a part-time job     | Binary      |
| `previous_grade`             | Previous academic grade/score               | Numerical   |
| `final_exam_score`           | Final examination score                     | Numerical   |
| `final_grade`                | Final grade to predict                      | Target      |

---

# 🎯 Target Variable

The target variable is:

```text
final_grade
```

The target contains the following grade categories:

```text
A
B
C
D
F
```

Because machine-learning classifiers work with numerical class labels, the target is transformed using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
```

---

# 🧹 Data Preprocessing

## 1. Removing Identifier

The `student_id` column is removed because it is an identifier rather than a meaningful predictive feature.

```python
df = df.drop('student_id', axis=1)
```

---

## 2. Binary Encoding

The following features contain `Yes`/`No` values:

```text
internet_access
extracurricular_activities
part_time_job
```

They are converted into numerical values:

```text
Yes → 1
No  → 0
```

Implementation:

```python
cols = [
    'internet_access',
    'extracurricular_activities',
    'part_time_job'
]

for col in cols:
    df[col] = df[col].map({
        'Yes': 1,
        'No': 0
    })
```

---

## 3. Feature and Target Separation

```python
X = df.drop('final_grade', axis=1)
y = df['final_grade']
```

Where:

* `X` contains the input features
* `y` contains the target grade

---

# ⚙️ Preprocessing Pipeline

The project automatically identifies numerical and categorical features.

### Numerical Features

```python
num_x = X.select_dtypes(include=np.number)
```

Numerical features are processed using:

```python
num_enc = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
```

This performs:

1. Missing-value imputation using the median
2. Feature standardization

---

### Categorical Features

```python
cat_x = X.select_dtypes(include='object')
```

Categorical features are processed using:

```python
cat_enc = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder())
])
```

This performs:

1. Missing-value imputation
2. One-hot encoding

---

## 🔗 ColumnTransformer

The different preprocessing strategies are combined using `ColumnTransformer`.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_enc, num_x.columns),
        ('cat', cat_enc, cat_x.columns)
    ]
)
```

This allows the same preprocessing workflow to be reused consistently during training and prediction.

---

# 🤖 Machine Learning Models

Several classification algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was used as the primary classification model.

```python
LogisticRegression(
    class_weight='balanced'
)
```

The complete pipeline:

```python
model = Pipeline(steps=[
    ('processing', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced'
    ))
])
```

Using `class_weight='balanced'` helps account for differences in class frequency.

---

## 2. Random Forest

Random Forest combines multiple decision trees to make predictions.

```python
RandomForestClassifier(
    class_weight='balanced'
)
```

It is useful for capturing nonlinear relationships between student characteristics and performance.

---

## 3. Gradient Boosting

Gradient Boosting builds models sequentially, with each new model attempting to improve the errors made by previous models.

```python
GradientBoostingClassifier()
```

---

## 4. Support Vector Machine

Support Vector Machine was also evaluated.

```python
SVC(
    C=0.1,
    gamma=10,
    class_weight='balanced'
)
```

SVM attempts to find effective decision boundaries between different grade classes.

---

## 5. K-Nearest Neighbors

KNN predicts the class of a student based on nearby observations.

```python
KNeighborsClassifier()
```

---

## 6. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm.

```python
GaussianNB()
```

---

## 7. XGBoost

XGBoost was included as a gradient-boosting model for comparison.

```python
XGBClassifier()
```

---

# 📈 Model Evaluation

The models are evaluated using multiple metrics.

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

Accuracy measures the percentage of correctly classified students.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(
    y_test,
    y_pred
)
```

This provides a more detailed understanding of model performance across individual grade classes.

---

### Confusion Matrix

A confusion matrix is generated to understand how predictions are distributed across the actual grade categories.

```python
confusion_matrix(
    y_test,
    y_pred
)
```

---

# 🔁 Cross-Validation

The Logistic Regression pipeline is evaluated using **5-fold cross-validation**.

```python
cv = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print(f"Cross-Validation Scores: {cv}")
print(f"Mean Accuracy: {cv.mean():.4f}")
```

### Why Cross-Validation?

Instead of evaluating the model using only one train/test split, 5-fold cross-validation repeatedly trains and validates the model on different portions of the dataset.

```text
                 Complete Dataset
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
      Fold 1          Fold 2          Fold 3
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                     Fold 4
                        │
                        ▼
                     Fold 5
                        │
                        ▼
              Average Performance
```

This provides a more robust estimate of model performance.

---

# 📋 Algorithms Compared

| Algorithm              | Category                     |
| ---------------------- | ---------------------------- |
| Logistic Regression    | Linear Classification        |
| Random Forest          | Ensemble Learning            |
| Gradient Boosting      | Ensemble Learning            |
| Support Vector Machine | Margin-Based Classification  |
| K-Nearest Neighbors    | Instance-Based Learning      |
| Gaussian Naive Bayes   | Probabilistic Classification |
| XGBoost                | Gradient Boosting            |

All models use the project's preprocessing pipeline where applicable, making the comparison more consistent.

---

# 🌐 Streamlit Web Application

The project includes an interactive **Streamlit** application for making predictions.

The application allows the user to enter:

### Student Information

* Gender
* Study time
* Previous grade
* Parental education
* Sleep hours
* Attendance percentage

### Academic Information

* Final examination score

### Activities & Access

* Internet access
* Extracurricular activities
* Part-time job

After entering the information, the user can click:

```text
🔮 Predict Final Grade
```

The application loads the trained model and displays the predicted grade.

Example:

```text
🎯 Predicted Final Grade: A
```

---

# 📸 Application Screenshot

A screenshot of the Streamlit application is available in:

```text
screenshots/
└── Screenshot 2026-09-05 193500.png
```

You can display it on GitHub using:

```markdown
![Student Performance Prediction App](screenshots/Screenshot%202026-09-05%20193500.png)
```

---

# 📁 Project Structure

```text
Student_Performance_Prediction/
│
├── app.py
│
├── student_performance.ipynb
│
├── student_performance_dataset.csv
│
├── trained_model.pkl
│
├── trained_model.joblib
│
├── requirements.txt
│
├── screenshots/
│   └── Screenshot 2026-09-05 193500.png
│
├── .gitignore
│
└── README.md
```

### File Description

| File                              | Purpose                            |
| --------------------------------- | ---------------------------------- |
| `app.py`                          | Streamlit prediction application   |
| `student_performance.ipynb`       | Complete machine-learning workflow |
| `student_performance_dataset.csv` | Dataset                            |
| `trained_model.pkl`               | Saved trained model                |
| `trained_model.joblib`            | Serialized model file              |
| `requirements.txt`                | Python dependencies                |
| `screenshots/`                    | Application screenshots            |
| `.gitignore`                      | Git ignored files                  |
| `README.md`                       | Project documentation              |

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Environment

* Jupyter Notebook / Google Colab
* VS Code
* Git & GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Student-Performance-Prediction.git
```

Navigate into the project:

```bash
cd Student-Performance-Prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

Streamlit will launch the application in your browser.

---

# 💻 Using the Application

1. Start the Streamlit application.
2. Enter the student's information.
3. Select the appropriate categorical values.
4. Adjust the numerical values using the sliders.
5. Click **Predict Final Grade**.
6. View the predicted final grade.

---

# 📓 Running the Jupyter Notebook

The complete machine-learning workflow is available in:

```text
student_performance.ipynb
```

The notebook contains:

```text
Data Loading
    ↓
Data Exploration
    ↓
Data Cleaning
    ↓
Visualization
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Train-Test Split
    ↓
Model Training
    ↓
Evaluation
    ↓
Cross-Validation
    ↓
Model Comparison
```

If using Google Colab, upload:

```text
student_performance.ipynb
student_performance_dataset.csv
```

Then execute the notebook cells sequentially.

---

# 📌 Requirements

The project uses the following major dependencies:

```text
streamlit
pandas==2.2.3
numpy==2.1.3
matplotlib
scikit-learn==1.6.1
joblib==1.5.3
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

# ⚠️ Model Compatibility

The saved model was created using the versions specified in `requirements.txt`.

For the most reliable results, use the same or compatible versions of:

```text
scikit-learn
numpy
pandas
joblib
```

When loading serialized Scikit-learn models, using significantly different Scikit-learn versions can cause compatibility issues.

---

# 🔮 Future Improvements

Potential improvements for future versions include:

* [ ] Hyperparameter tuning using `GridSearchCV`
* [ ] Randomized hyperparameter search
* [ ] Improved model comparison
* [ ] ROC-AUC analysis
* [ ] Interactive confusion matrix
* [ ] Feature importance visualization
* [ ] Prediction probability display
* [ ] Better handling of unseen categorical values
* [ ] Model versioning
* [ ] Automated model retraining
* [ ] Streamlit Cloud deployment
* [ ] Docker support
* [ ] Improved UI styling
* [ ] Explainable AI using SHAP

---

# 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Numerical scaling
* Missing-value handling
* Scikit-learn pipelines
* ColumnTransformer
* Classification algorithms
* Ensemble learning
* Model evaluation
* Cross-validation
* Model serialization
* Streamlit application development
* Machine-learning deployment workflow

---

# 📜 Disclaimer

This project is intended for **educational and demonstration purposes**.

The predicted grade should not be considered an official academic assessment or a substitute for evaluation by teachers or educational institutions.

---

# 👨‍💻 Author

**Jilla SriVardhan**

This project was created as part of a practical machine-learning and Streamlit development portfolio.

---

## ⭐ If You Found This Project Useful

If you found this project helpful or interesting, consider giving the repository a ⭐ on GitHub.

Contributions, suggestions, and improvements are welcome!
