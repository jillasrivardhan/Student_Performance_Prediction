# 🎓 Student Performance Prediction

A machine learning classification project that predicts a student's **final grade category** based on academic, behavioral, and lifestyle-related factors.

The project demonstrates an end-to-end machine learning workflow using **Python and Scikit-learn**, including data preprocessing, exploratory data analysis, categorical encoding, feature scaling, Logistic Regression, model evaluation, cross-validation, and comparison of multiple classification algorithms.

---

## 📌 Project Overview

Student performance can be influenced by several factors, including attendance, study habits, sleep, extracurricular activities, internet access, part-time employment, and examination performance.

The objective of this project is to use these student-related features to build machine learning models capable of predicting the student's **final grade category**.

### Machine Learning Workflow

```text
Student Performance Dataset
          ↓
     Data Loading
          ↓
  Data Exploration
          ↓
    Data Cleaning
          ↓
   Feature Encoding
          ↓
 Feature / Target Split
          ↓
 Train-Test Split
          ↓
Preprocessing Pipeline
          ↓
  Model Training
          ↓
    Prediction
          ↓
 Model Evaluation
          ↓
 Cross-Validation
          ↓
Multiple Model Comparison
```

---

## 🎯 Objectives

* Explore the student performance dataset
* Understand the available academic and behavioral features
* Check for missing values
* Remove unnecessary identifier columns
* Convert binary categorical features into numerical values
* Separate features and target variable
* Encode the target variable
* Automatically identify numerical and categorical features
* Handle missing values using imputation
* Scale numerical features
* One-hot encode categorical features
* Build a reusable Scikit-learn preprocessing pipeline
* Train a Logistic Regression classifier
* Evaluate the model using multiple classification metrics
* Perform 5-fold cross-validation
* Compare multiple machine learning algorithms

---

## 📊 Dataset

The project uses:

```text
student_performance_dataset.csv
```

### Dataset Features

The dataset contains student-related information used to predict the final grade.

| Feature                      | Description                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| `student_id`                 | Unique identifier for each student                              |
| `internet_access`            | Whether the student has internet access                         |
| `extracurricular_activities` | Whether the student participates in extracurricular activities  |
| `part_time_job`              | Whether the student has a part-time job                         |
| `sleep_hours`                | Number of hours the student sleeps                              |
| `attendance_percent`         | Student attendance percentage                                   |
| `final_exam_score`           | Student's final examination score                               |
| `final_grade`                | Target variable representing the student's final grade category |

> The exact feature distribution and categories are determined by the supplied dataset.

---

## 🎯 Target Variable

The target variable is:

```text
final_grade
```

Since `final_grade` contains categorical values, it is converted into numerical labels using **LabelEncoder** before model training.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
```

This allows classification algorithms to work with the target variable.

---

# 🔍 Exploratory Data Analysis

The notebook performs several exploratory analysis steps to understand the dataset.

### Dataset Preview

```python
df.head()
```

### Dataset Information

```python
df.info()
```

### Missing Value Analysis

```python
df.isnull().sum()
```

These steps help understand the dataset structure, data types, and potential missing values.

---

## 📈 Visualizations

The project uses **Seaborn** to visualize relationships between student characteristics and performance.

### Sleep Hours vs Final Grade

A count plot is used to examine the relationship between sleep hours and final grade.

```python
sns.countplot(
    x='sleep_hours',
    data=df,
    hue='final_grade'
)
```

### Sleep Hours vs Attendance

A scatter plot is used to visualize the relationship between sleep hours and attendance.

```python
sns.scatterplot(
    x='sleep_hours',
    y='attendance_percent',
    data=df
)
```

### Extracurricular Activities vs Final Grade

A count plot is used to compare final grades based on extracurricular participation.

```python
sns.countplot(
    x='extracurricular_activities',
    hue='final_grade',
    data=df
)
```

These visualizations provide an initial understanding of patterns within the dataset.

---

# 🧹 Data Preprocessing

## 1. Removing the Student ID

`student_id` is an identifier rather than a meaningful predictive feature, so it is removed:

```python
df = df.drop('student_id', axis=1)
```

This prevents the model from using an arbitrary identifier during training.

---

## 2. Encoding Binary Features

The following columns contain `Yes`/`No` values:

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

Using:

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

## 3. Separating Features and Target

The target variable is separated from the input features:

```python
X = df.drop('final_grade', axis=1)
y = df['final_grade']
```

Where:

* `X` → Input features
* `y` → Target variable

---

## 4. Target Encoding

The categorical target is transformed using `LabelEncoder`:

```python
le = LabelEncoder()

y = le.fit_transform(y)
```

This converts the grade categories into numerical class labels.

---

# ⚙️ Feature Preprocessing Pipeline

The project automatically separates numerical and categorical features.

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
2. Standardization using `StandardScaler`

---

### Categorical Features

Categorical features are identified using:

```python
cat_x = X.select_dtypes(include='object')
```

They are processed using:

```python
cat_enc = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder())
])
```

This performs:

1. Missing-value imputation using the most frequent value
2. One-hot encoding of categorical variables

---

## 🧩 ColumnTransformer

Different preprocessing strategies are applied to numerical and categorical columns using `ColumnTransformer`.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_enc, num_x.columns),
        ('cat', cat_enc, cat_x.columns)
    ]
)
```

This creates a clean and reusable preprocessing workflow.

---

# 🤖 Machine Learning Models

The project starts with **Logistic Regression** as the primary classification model and then compares it with several other algorithms.

## 1. Logistic Regression

The primary model is:

```python
LogisticRegression(
    class_weight='balanced'
)
```

The complete pipeline is:

```python
model = Pipeline(steps=[
    ('processing', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced'
    ))
])
```

`class_weight='balanced'` helps compensate when different target classes have unequal frequencies.

---

# 🌲 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

```python
RandomForestClassifier(
    class_weight='balanced'
)
```

It can capture nonlinear relationships between student characteristics and performance.

---

# 📈 3. Gradient Boosting

Gradient Boosting builds models sequentially, where each new model attempts to improve upon previous errors.

```python
GradientBoostingClassifier()
```

It is useful for learning complex relationships in structured datasets.

---

# 🧠 4. Support Vector Machine

The project also evaluates a Support Vector Machine:

```python
SVC(
    C=0.1,
    gamma=10,
    class_weight='balanced'
)
```

SVM attempts to find decision boundaries that effectively separate different classes.

---

# 📍 5. K-Nearest Neighbors

KNN predicts the class of a sample based on nearby observations.

```python
KNeighborsClassifier()
```

It is a simple but useful classification algorithm for comparison.

---

# 🧮 6. Naive Bayes

The project includes:

```python
GaussianNB()
```

Gaussian Naive Bayes assumes that numerical features follow a Gaussian distribution within each class.

---

# 🚀 7. XGBoost

The project also experiments with XGBoost:

```python
XGBClassifier()
```

XGBoost is a powerful gradient-boosting algorithm commonly used for structured/tabular machine learning problems.

---

# 🔗 Model Pipeline

Each classification algorithm is combined with the same preprocessing pipeline:

```python
model_pipeline = Pipeline(steps=[
    ('processing', preprocessor),
    ('model', algo)
])
```

This ensures that preprocessing is performed consistently before the model receives the data.

---

# 📊 Model Evaluation

The project evaluates the classification models using:

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

Accuracy represents the proportion of correctly classified observations.

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

These metrics provide more detailed information than accuracy alone.

---

### Confusion Matrix

The confusion matrix shows how predictions are distributed across the actual classes.

```python
confusion_matrix(
    y_test,
    y_pred
)
```

It helps identify which classes the model predicts correctly and which classes it confuses.

---

# 🔁 5-Fold Cross-Validation

The Logistic Regression pipeline is also evaluated using **5-fold cross-validation**.

```python
cv = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)
```

The individual scores and mean accuracy are calculated:

```python
print(f"Cross-Validation Scores: {cv}")
print(f"Mean Accuracy: {cv.mean():.4f}")
```

### How 5-Fold Cross-Validation Works

```text
             Dataset
                │
      ┌─────────┼─────────┐
      ↓         ↓         ↓
    Fold 1    Fold 2    Fold 3
      ↓         ↓         ↓
    Fold 4    Fold 5
                │
                ↓
        Average Performance
```

The dataset is divided into five parts. The model is trained and validated multiple times using different portions of the data.

This provides a more reliable estimate of model performance than relying on a single train-test split.

---

# 📋 Model Comparison

The project compares the following algorithms:

| Model                  | Type                         |
| ---------------------- | ---------------------------- |
| Logistic Regression    | Linear Classification        |
| Random Forest          | Ensemble Learning            |
| Gradient Boosting      | Ensemble Learning            |
| Support Vector Machine | Margin-Based Classification  |
| K-Nearest Neighbors    | Instance-Based Learning      |
| Naive Bayes            | Probabilistic Classification |
| XGBoost                | Gradient Boosting            |

Each model uses the same preprocessing pipeline, making the comparison more consistent.

---

# 🗂️ Project Structure

```text
Student_Performance_Prediction/
│
├── student_performance.ipynb
├── student_performance_dataset.csv
└── README.md
```

---

# 🚀 How to Run the Project

## Option 1 — Google Colab

The notebook is designed to work well with Google Colab.

### Step 1 — Open the Notebook

Upload:

```text
student_performance.ipynb
```

to Google Colab.

### Step 2 — Upload the Dataset

Upload:

```text
student_performance_dataset.csv
```

to the Colab environment.

### Step 3 — Install XGBoost

If XGBoost is not already available:

```python
!pip install xgboost
```

### Step 4 — Run the Notebook

Run the cells sequentially from top to bottom.

---

## Option 2 — Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/jillasrivardhan/Student-Performance-Prediction.git
```

Navigate into the project:

```bash
cd Student-Performance-Prediction
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
student_performance.ipynb
```

and execute the cells sequentially.

---

# 📦 Requirements

The project uses the following Python libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install everything with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project provides practical experience with:

* Classification
* Logistic Regression
* Ensemble Learning
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Machines
* K-Nearest Neighbors
* Naive Bayes
* Exploratory Data Analysis
* Data Cleaning
* Feature Engineering
* Label Encoding
* One-Hot Encoding
* Missing Value Imputation
* Feature Scaling
* `StandardScaler`
* `SimpleImputer`
* `ColumnTransformer`
* `Pipeline`
* Train-Test Split
* Cross-Validation
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Model Comparison

---

# 💡 Key Learnings

Through this project, I practiced building a complete classification workflow from raw data to model comparison.

### Main takeaways

* Understanding how student-related features can be used for classification
* Handling categorical variables
* Encoding binary features
* Encoding categorical target variables
* Handling missing values
* Scaling numerical features
* Building reusable preprocessing pipelines
* Using `ColumnTransformer`
* Training multiple classification algorithms
* Evaluating classification performance
* Applying cross-validation
* Comparing different machine learning approaches

---

# 🚧 Future Improvements

The project can be extended with:

* Hyperparameter tuning using `GridSearchCV`
* `RandomizedSearchCV`
* Feature importance analysis
* ROC-AUC evaluation
* Precision-Recall curves
* Model performance visualization
* Hyperparameter optimization for XGBoost
* Class distribution analysis
* More advanced ensemble techniques
* Explainable AI using SHAP
* Saving the best model using Joblib
* Building a Streamlit prediction interface
* Deploying the application as a web service

---

# ⚠️ Disclaimer

This project is intended for **educational and machine learning practice purposes**.

Student performance is influenced by many academic, personal, social, and environmental factors. Predictions generated by this model should not be treated as definitive assessments of a student's abilities or future performance.

---

# 👨‍💻 Author

**Jilla Srivardhan**

GitHub: [Jilla Srivardhan](https://github.com/jillasrivardhan?utm_source=chatgpt.com)

---

# ⭐ Support

If you found this project useful for learning machine learning and classification, consider giving the repository a ⭐.

```text
Data → Preprocessing → Feature Engineering → Classification
                         ↓
              Model Evaluation → Cross-Validation
                         ↓
                  Model Comparison 🚀
```
