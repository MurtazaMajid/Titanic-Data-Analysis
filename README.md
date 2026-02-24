# Titanic Survival Prediction | End-to-End ML Analysis

This repository explores the Titanic passenger dataset to understand factors affecting survival and builds a **Logistic Regression model** to predict passenger survival. The project follows a complete data science workflow including **data cleaning, feature engineering, EDA, and predictive modeling**.

The final model achieves **~76% accuracy** using a small set of meaningful features.

---

## Overview | Highlights

Data Cleaning | Feature Engineering | Exploratory Data Analysis | Logistic Regression | Model Evaluation

**Key Points:**

- End-to-end machine learning workflow  
- Analysis of survival patterns by gender, class, age, and family size  
- Logistic Regression model achieving 76% accuracy  
- Clear and reproducible workflow for any user

---

## Dataset | Source

**Source:** Kaggle Titanic Dataset  
**Total Passengers:** 891  

| Feature         | Description                           |
|-----------------|---------------------------------------|
| PassengerId     | Unique passenger identifier           |
| Pclass          | Passenger class (1st, 2nd, 3rd)      |
| Name            | Name of passenger                     |
| Sex             | Gender                                |
| Age             | Age of passenger                      |
| SibSp           | # of siblings/spouses aboard          |
| Parch           | # of parents/children aboard          |
| Ticket          | Ticket number                         |
| Fare            | Ticket fare                           |
| Cabin           | Cabin number                           |
| Embarked        | Port of embarkation                   |
| Survived        | Target variable (0 = No, 1 = Yes)    |

---

## Project Workflow | Steps

Data Loading | Cleaning | Missing Values | Outlier Detection | Feature Engineering | EDA | Model Training | Model Evaluation

1. Load the dataset  
2. Remove irrelevant columns (`Name`, `Ticket`, `Cabin`)  
3. Handle missing values (`Age` via linear interpolation)  
4. Remove outliers in `Fare` using IQR method  
5. Encode categorical variables (`Sex`)  
6. Create new features (`Age Groups`, `Family Size`)  
7. Perform EDA and visualize survival patterns  
8. Train a Logistic Regression model using `Age`, `Pclass`, `Sex`  
9. Evaluate model performance (accuracy, confusion matrix, classification report)  

---

## Exploratory Data Analysis | Key Insights

| Analysis                       | Finding                                         |
|--------------------------------|------------------------------------------------|
| Survival by Gender              | Female: 68.9%, Male: 17.9%                    |
| Survival by Passenger Class     | 1st: 50.9%, 2nd: 48.6%, 3rd: 24.6%           |
| Survival by Age Group           | Children highest, Teens lowest                 |
| Survival by Family Size         | Families of 3 had highest survival (72.7%)    |
| Embarkation Point               | C: 44.8%, Q: 38.7%, S: 31.2%                  |

**Observation:** Gender and passenger class had the strongest impact on survival. Small families also fared better.

---

## Machine Learning Model | Logistic Regression

**Features Used:** `Age`, `Pclass`, `Sex`

### Train/Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Model Training & Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(conf_matrix)
print(class_report)
```

**Performance Summary:**

| Metric           | Score           |
|------------------|----------------|
| Accuracy         | 0.76           |
| Precision (0)    | 0.80           |
| Recall (0)       | 0.85           |
| Precision (1)    | 0.68           |
| Recall (1)       | 0.61           |

---

## Example Prediction

| Passenger Details       | Prediction        | Probability |
|-------------------------|-----------------|-------------|
| Age: 21, Male, 1st Class | Survived         | 58.5%      |

---

## Repository Structure

```
Titanic-Data-Analysis
│
├── notebooks/
│   └── titanic.ipynb         # Main analysis notebook
├── data/
│   └── titanic_cleaned.csv   # Cleaned dataset
├── images/                   # Plots and visualizations
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation | Run Project

Clone the repository:

```bash
git clone https://github.com/MurtazaMajid/Titanic-Data-Analysis.git
cd Titanic-Data-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/titanic.ipynb
```

---

## Skills Demonstrated

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn | Data Cleaning | EDA | Feature Engineering | Logistic Regression | Model Evaluation

---

## Future Improvements

- Include additional features such as passenger titles or cabin info  
- Test advanced models like Random Forest, XGBoost, SVM  
- Hyperparameter tuning with GridSearchCV  
- Cross-validation for robust evaluation  
- Deploy the model as a web application

---

## Author

**Murtaza Majid**  
GitHub: [@MurtazaMajid](https://github.com/MurtazaMajid)
