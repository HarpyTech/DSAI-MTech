# Performance Optimization in Logistic Regression Using Parameter Tuning

## Introduction

Supervised classification is a fundamental machine learning task where a model learns from labeled data to classify new instances into predefined categories. Several classification algorithms exist, including Logistic Regression, Decision Trees, K-Nearest Neighbors, Naïve Bayes, and ensemble techniques like Random Forest and Boosting. Evaluating these models requires performance measures like accuracy, precision, recall, F1-score, and ROC-AUC.

This blog explores parameter tuning techniques to optimize the performance of Logistic Regression, Decision Tree, and Random Forest models. Additionally, we discuss handling imbalanced datasets and using advanced ensemble techniques to enhance predictive accuracy, along with code examples and real-time applications. Understanding these methods is essential for improving classification performance in real-world scenarios, such as financial risk assessment, fraud detection, and medical diagnosis.

## Real-World Application: Predicting Loan Default

To better understand these models, let's consider a real-world problem: predicting whether a customer will default on a loan. This involves classifying customers as **defaulters (1) or non-defaulters (0)** based on features like income, credit score, loan amount, and payment history. Accurate predictions can help financial institutions mitigate risks and improve lending strategies.

### Understanding Classification Models

Before delving into optimization techniques, it's essential to grasp the working mechanisms of key classification models:
- **Logistic Regression**: Uses a linear function and applies a sigmoid activation to classify outcomes probabilistically.
- **Decision Trees**: Splits data recursively based on feature importance to make classification decisions.
- **Random Forest**: An ensemble method that aggregates multiple decision trees for robust classification.
- **Boosting Methods**: Focus on iteratively improving weak models by prioritizing difficult-to-classify instances.

## Logistic Regression and Odd's Probability with Performance Measures

Logistic Regression is a linear model used for binary classification. It predicts probabilities using the sigmoid function:

$$
P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \ldots + \beta_nX_n)}}
$$


### Performance Measures:

- **Accuracy**: Correct predictions over total predictions.
- **Precision & Recall**: Measures quality and sensitivity.
- **F1-score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Evaluates discrimination ability between classes.

#### Parameter Tuning:

- **Regularization (L1 & L2)**: Helps prevent overfitting.
- **Solver selection (liblinear, saga, lbfgs)**: Impacts optimization convergence.
- **C (Inverse Regularization Strength)**: Controls model complexity.

### Effect of Hyperparameters

Regularization techniques, like **L1 (Lasso)** and **L2 (Ridge)**, help in preventing overfitting and feature selection. Choosing an optimal regularization parameter (**C**) directly impacts model performance. A small C value forces stronger regularization, reducing model complexity but potentially underfitting the data. A large C value relaxes regularization, which may lead to overfitting.

#### Implementation in Python (Loan Default Prediction Example):

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
loan_data = pd.read_csv('loan_default_data.csv')
X = loan_data[['income', 'credit_score', 'loan_amount', 'payment_history']]
y = loan_data['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

log_reg = LogisticRegression()
grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Model evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Handling Imbalanced Datasets with SMOTE & Random Under Sampling

Many real-world datasets, such as fraud detection and medical diagnosis, are highly imbalanced. An imbalanced dataset can lead to biased predictions, where the model favors the majority class. To counteract this, we can use **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class or use **random under-sampling** to reduce the number of majority class instances.

### Implementation:

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Importance of Handling Imbalanced Data

Failing to address imbalanced datasets can lead to misleading model evaluations. Metrics like **accuracy** can be deceptive when one class is predominant. Techniques such as **Precision-Recall tradeoff** and **ROC-AUC analysis** are crucial in such cases.

## Bagging Classifier & Random Forest with Hyperparameter Tuning (Customer Loan Approval)

Bagging (Bootstrap Aggregating) reduces variance by training multiple base classifiers on bootstrapped data samples. Random Forest extends this by training multiple decision trees and averaging their predictions for better performance.

### Random Forest Implementation:

```python
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier()
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
```

## Boosting, Stacking, and Voting in Ensemble Learning

Boosting techniques combine weak learners to build strong models by focusing more on misclassified instances in each iteration. Boosting methods include **AdaBoost, Gradient Boosting, and XGBoost.**

### Stacking & Voting:

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('svm', SVC(probability=True))],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

## Conclusion

Fine-tuning classifiers is crucial for performance improvement. Using real-world examples like **loan default prediction, fraud detection, and loan approval risk analysis**, we demonstrated how parameter tuning can significantly improve model accuracy. Techniques like **SMOTE for imbalanced data, Decision Trees for risk profiling, and ensemble methods like Boosting and Stacking** provide robust solutions to real-world classification problems. By applying these methodologies, businesses can improve decision-making processes, reduce risks, and enhance predictive model accuracy in dynamic environments.




--- 

To demonstrate the application of various parameter tuning and optimization techniques, we'll use the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository. This dataset is commonly used for binary classification tasks, where the goal is to predict whether a tumor is malignant or benign based on various features.

**Dataset Overview:**

- **Features:** 30 numeric features representing characteristics of cell nuclei present in images.
- **Target:** Binary classification—Malignant (1) or Benign (0).
- **Size:** 569 samples.

**Steps to Apply Tuning Methods:**

1. **Data Loading and Preprocessing:**
   - Load the dataset.
   - Handle missing values (if any).
   - Split the data into training and testing sets.

2. **Baseline Model Training:**
   - Train a Logistic Regression model without parameter tuning.
   - Evaluate performance using metrics like accuracy, precision, recall, and F1-score.

3. **Hyperparameter Tuning:**
   - **Logistic Regression:** Tune regularization strength and solver.
   - **Decision Tree:** Tune criteria (Gini/Entropy), maximum depth, and minimum samples split.
   - **Random Forest:** Tune the number of estimators, maximum features, and bootstrap options.
   - **K-Nearest Neighbors (KNN):** Tune the number of neighbors.
   - **Naïve Bayes:** Although it has fewer hyperparameters, consider variations like Gaussian, Multinomial, or Bernoulli based on data characteristics.

4. **Handling Imbalanced Datasets:**
   - Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance classes if the dataset is imbalanced.
   - Alternatively, use random under-sampling to reduce the majority class.

5. **Ensemble Methods:**
   - Implement Bagging Classifier.
   - Use Boosting techniques like AdaBoost, Gradient Boosting, and XGBoost.
   - Combine models using Stacking and Voting classifiers.

6. **Model Evaluation:**
   - Compare the performance of all models using cross-validation.
   - Select the best-performing model based on evaluation metrics.

**Implementation in Python:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMote
from xgboost import XGBClassifier

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Logistic Regression
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Baseline Logistic Regression:\n", classification_report(y_test, y_pred))

# Hyperparameter Tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_log_reg = grid.best_estimator_
y_pred = best_log_reg.predict(X_test)
print("Tuned Logistic Regression:\n", classification_report(y_test, y_pred))

# Decision Tree with Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)
print("Tuned Decision Tree:\n", classification_report(y_test, y_pred))

# Random Forest with Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
print("Tuned Random Forest:\n", classification_report(y_test, y_pred))

# K-Nearest Neighbors with Hyperparameter Tuning
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
print("Tuned K-Nearest Neighbors:\n", classification_report(y_test, y_pred))

# Naïve Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Naïve Bayes:\n", classification_report(y_test, y_pred))

# Handling Imbalanced Dataset with SMOTE (if needed)
# Check class distribution
print("Class distribution before SMOTE:", y_train.value_counts())
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", pd.Series(y_resampled).value_counts())

# Ensemble Methods
# AdaBoost
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
print("AdaBoost:\n", classification_report(y_test, y_pred))

# Gradient Boosting
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
print("Gradient Boosting:\n", classification_report(y_test, y_pred))

# XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("XGBoost:\n", classification_report(y_test, y_pred))

# Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svm', SVC(probability=True))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print("Stacking Classifier:\n", classification_report(y_test, y_pred))

# Voting Classifier
voting::contentReference[oaicite:0]{index=0} 
```




---

# Automating Employee Attrition Prediction with Scikit-Learn Pipelines

Employee attrition prediction is an essential task for organizations aiming to reduce turnover and retain top talent. The ability to preprocess data efficiently plays a crucial role in model performance, interpretability, and the overall success of predictive modeling. Manual preprocessing, however, can be time-consuming, prone to errors, and difficult to reproduce. By leveraging Scikit-Learn’s `Pipeline`, `ColumnTransformer`, and custom transformers, we can automate the entire preprocessing workflow, ensuring consistency, scalability, and efficiency in data transformation.

## Why Use Transformers and Pipelines?

Using Scikit-Learn transformers and pipelines provides numerous advantages, including:

- **Consistency**: Ensures identical preprocessing steps are applied to both training and inference data, eliminating discrepancies.
- **Modularity**: Enables easy swapping of preprocessing components and machine learning models.
- **Efficiency**: Reduces redundant code, speeds up experimentation, and minimizes the risk of data leakage.
- **Scalability**: Handles large datasets effectively and integrates seamlessly into automated ML workflows.
- **Reproducibility**: Provides a structured, repeatable framework that can be easily shared and deployed.

You can check out the full implementation of this pipeline in my GitHub repository: [Full Code Here](https://github.com/HarpyTech/transformers/blob/main/ML2_Hackathon.ipynb). This will give you deeper insights into how the pipeline is structured and how it can be further customized for your specific use case.

## Employee Attrition Prediction Pipeline

The `EmployeeAttritionPipeline` class automates preprocessing and model selection, allowing seamless integration of different classifiers while keeping the data transformation process standardized.

### Required Libraries

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
```

### Custom Employee Attrition Pipeline

The following Python class defines an end-to-end machine learning pipeline for predicting employee attrition. It includes:
- **Data Preprocessing**: Handling missing values, scaling numerical features, and encoding categorical variables.
- **Outlier Handling**: Ensuring extreme values do not skew model performance.
- **Model Selection**: Allowing flexibility to use different classifiers like logistic regression, decision trees, or random forests.
- **Cross-Validation**: Evaluating model performance across multiple folds of data.

```python
class EmployeeAttritionPipeline:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.pipeline = None

    def create_pipeline(self, numerical_cols, categorical_cols):
        numerical_transformer = Pipeline(steps=[
            ('imputer', MissingValueImputer(strategy='median')),
            ('outlier_handler', OutlierTransformation()),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', MissingValueImputer(categorical_strategy='mode')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier()
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier()
        else:
            raise ValueError("Invalid model type. Choose 'logistic_regression', 'random_forest', or 'decision_tree'.")

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', self.model)])
        return self.pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy}")
        print(classification_report(y, y_pred))
        return accuracy

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        print(f"Cross Validation Scores: {scores}")
        print(f"Average Cross-Validation Score: {scores.mean()}")
```

## Applying the Pipeline to Employee Attrition Data

```python
numerical_cols = numerical_columns[2:]
categorical_cols = categorical_columns

X = data.drop(columns=['Attrition', 'EmployeeID'], axis=1)
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the pipeline
pipeline = EmployeeAttritionPipeline(model_type='logistic_regression') # Change model type here
pipeline.create_pipeline(numerical_cols, categorical_cols)
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
pipeline.evaluate(X_test, y_test)

pipeline.cross_validate(X, y)
```

## Benefits of Using Transformers in Preprocessing

### 1. **Handles Missing Data Automatically**
Using `MissingValueImputer`, we replace missing values with the median (for numerical data) and mode (for categorical data), ensuring robust data quality without dropping records.

### 2. **Manages Outliers Effectively**
`OutlierTransformation` helps prevent extreme values from skewing model performance, leading to better generalization and more stable predictions.

### 3. **Scales Numerical Features**
`StandardScaler` ensures that numerical columns have zero mean and unit variance, improving the efficiency of gradient-based models like logistic regression and neural networks.

### 4. **Encodes Categorical Variables**
`OneHotEncoder` converts categorical features into a numerical format, handling unknown categories gracefully and preserving useful information.

### 5. **Prevents Data Leakage**
By encapsulating preprocessing steps within a pipeline, we avoid data leakage, where information from the test set inadvertently influences training.

### 6. **Supports Multiple Models Seamlessly**
The structured design of the `EmployeeAttritionPipeline` allows easy experimentation with different machine learning models, enabling data scientists to optimize performance efficiently.

### 7. **Enhances Reproducibility and Deployment**
The modularity of the pipeline makes it easier to save, share, and deploy the trained model in production environments, ensuring consistency across different datasets and use cases.

## Conclusion

Automating data preprocessing with Scikit-Learn pipelines and transformers provides a structured, efficient, and reproducible approach to machine learning. By integrating transformation steps within a unified framework, the `EmployeeAttritionPipeline` enhances model performance while reducing manual effort. Organizations can benefit from improved predictive accuracy, reduced human error, and streamlined deployment of machine learning solutions.

Would you like additional enhancements, such as hyperparameter tuning, feature selection, or advanced model stacking? Let me know how I can further refine this workflow for your use case!

