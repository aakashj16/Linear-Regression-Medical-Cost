# Linear-Regression-Medical-Cost
A comprehensive project on predicting medical costs using multiple regression techniques, including linear, ridge, and lasso regression.

## Project Overview

The dataset used in this project contains information about individuals' medical costs based on their age, gender, BMI, number of children, smoking status, and region. The goal is to use machine learning models to predict the medical charges (`charges`) based on these features.

The following models are implemented:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**

### Key Features:
- Age
- Sex
- BMI (Body Mass Index)
- Number of Children
- Smoker (Yes/No)
- Region

## Steps Performed

### 1. Data Exploration and Preprocessing:
- Read the dataset and performed data inspection (checking for null values, duplicates, and data types).
- Explored the distribution of categorical features (e.g., gender, smoker, and region) and visualized the relationships between numerical features (e.g., age, BMI, and medical charges).
- Removed outliers and treated skewness in features.
- One-hot encoded categorical features and performed scaling.

### 2. Feature Selection:
- Used **Variance Inflation Factor (VIF)** to identify and drop highly correlated features to avoid multicollinearity.
- Applied **Recursive Feature Elimination (RFE)** to select the most significant features for model building.

### 3. Model Building:
- Implemented three regression models to predict medical charges:
  - **Linear Regression**
  - **Ridge Regression**
  - **Lasso Regression**
- Evaluated the models using R2 score on both training and testing datasets.

### 4. Results:
- Linear Regression achieved an R2 score of 76.33% on the testing set.
- Ridge Regression outperformed with an R2 score of 83.64%.
- Lasso Regression provided competitive results with an R2 score of 82.8%.

## Requirements

To run the code, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`

You can install the dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels