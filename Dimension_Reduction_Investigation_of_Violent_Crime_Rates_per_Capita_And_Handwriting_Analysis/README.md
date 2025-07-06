# Investigation of Violent Crime Rates per Capita

## Project Overview

This project investigates violent crime rates per capita using advanced machine learning techniques, including kernel ridge regression, regularized linear models, and various classification algorithms. The analysis covers both regression problems (crime rate prediction) and classification problems (MNIST digit recognition).

## Table of Contents

- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Analysis Components](#analysis-components)
- [Results Summary](#results-summary)
- [Technical Implementation](#technical-implementation)
- [Dependencies](#dependencies)

## Data Sources

### Communities and Crime Dataset
- **Source**: UCI Machine Learning Repository (ID: 183)
- **Description**: Contains socio-economic and law enforcement data for communities across the United States
- **Target Variable**: Violent crime rate per capita
- **Features**: 122 features including demographic, economic, and law enforcement variables

### MNIST Dataset
- **Source**: Kaggle (hojjatk/mnist-dataset)
- **Description**: Handwritten digit recognition dataset
- **Subset**: Focused on digits 3, 5, and 8 for multi-class classification
- **Features**: 784 pixel values (28×28 images)

## Project Structure

```
Investigation_of_Violent_Crime_Rates_per_Capita/
├── README.md                 # This file
├── code.ipynb               # Main analysis notebook
├── Brief_Introduction.pdf   # Project introduction
├── Code_Part.pdf           # Code documentation
└── target.pdf              # Target specifications
```

## Analysis Components

### 1. Crime Rate Prediction (Regression Analysis)

#### Data Preprocessing
- **Missing Value Handling**: Replaced "?" values with NaN and used median imputation
- **Data Type Correction**: Converted object-type continuous variables to float32
- **Feature Engineering**: Created dummy variables for state categorical features
- **Data Splitting**: 80% training, 20% test split with random state 89

#### Kernel Ridge Regression Implementation
- **Custom Implementation**: Built polynomial and RBF kernel functions from scratch
- **Cross-Validation**: 5-fold CV for hyperparameter tuning
- **Kernel Types**:
  - **Polynomial Kernel**: Degree 3-5, coef0=1
  - **RBF Kernel**: Gamma values 0.01-10
- **Regularization**: Alpha values 0.01-100

#### Results
- **Polynomial Kernel**: Best MSE = 0.02175 (α=100, degree=3)
- **RBF Kernel**: Best MSE = 0.05401 (α=0.01, γ=1)
- **Final Test MSE**: 0.01791 (Polynomial kernel)

#### Linear Models Comparison
- **Ridge Regression**: Best MSE = 0.01855 (α=100)
- **Lasso Regression**: Tested α values 0.01-100
- **ElasticNet**: Tested α values 0.01-100, l1_ratio 0.1-0.9
- **Linear Regression**: Baseline without regularization

### 2. MNIST Digit Classification

#### Data Preparation
- **Filtered Classes**: Digits 3, 5, and 8 for multi-class classification
- **Feature Scaling**: StandardScaler applied to pixel values
- **Data Split**: Pre-split train/test datasets

#### Classification Models

##### 1. Logistic Regression
- **One-vs-Rest (OvR)**: Accuracy = 92.84%
- **Multinomial**: Accuracy = 92.84%
- **Regularization**: L1 (saga solver) and L2 (lbfgs solver)

##### 2. Naive Bayes
- **Model**: GaussianNB
- **Accuracy**: 35.29%
- **Performance**: Poor due to assumption violations

##### 3. Linear Discriminant Analysis (LDA)
- **Accuracy**: 90.44%
- **Performance**: Good for linear separability

##### 4. Support Vector Machine
- **Model**: LinearSVC with One-vs-Rest
- **Best C**: 0.001
- **Accuracy**: 92.73%

##### 5. Group Lasso
- **Model**: LogisticGroupLasso
- **Accuracy**: 92.63%
- **Feature Sparsity**: 32.14% features zeroed out
- **Advantage**: Automatic feature selection

#### Model Performance Comparison

| Model | Accuracy | Hardest Class |
|-------|----------|---------|
| Logistic Regression (OvR) | 92.84%| 3 |
| Logistic Regression (Multinomial) | 92.84%  | 3 |
| Linear SVM (OvR) | 92.73%| 5 |
| Group Lasso | 92.63% | 5 |
| LDA | 90.44% | 5 |
| Naive Bayes | 35.29%  | 5 |

## Results Summary

### Crime Rate Prediction
- **Best Model**: Polynomial Kernel Ridge Regression
- **Validation MSE**: 0.02175
- **Test MSE**: 0.01791
- **Key Insight**: Non-linear relationships captured effectively by polynomial kernel

### MNIST Classification
- **Best Model**: Logistic Regression (both OvR and Multinomial)
- **Accuracy**: 92.84%
- **Hardest Class**: Digit 5 (most frequently misclassified)
- **Key Insight**: Linear models perform well on MNIST, suggesting good linear separability

## Technical Implementation

### Key Libraries Used
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, group-lasso
- **Visualization**: matplotlib, seaborn
- **Data Sources**: ucimlrepo, kagglehub

### Custom Implementations
- **Kernel Functions**: Polynomial and RBF kernels from scratch
- **Kernel Ridge Regression**: Custom solver with matrix operations
- **Cross-Validation**: K-fold CV with proper train/validation splits

### Hyperparameter Tuning
- **Grid Search**: Systematic exploration of parameter spaces
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Regularization**: L1, L2, and elastic net penalties

## Dependencies

```python
# Core ML libraries
scikit-learn>=1.0.0
pandas>=1.0.0
numpy>=1.20.0

# Specialized libraries
group-lasso>=1.5.0
ucimlrepo>=0.0.7
kagglehub

# Visualization
matplotlib>=3.0.0
seaborn>=0.11.0

# Data handling
scipy>=1.6.0
```

## Key Findings

1. **Kernel Methods**: Polynomial kernels outperform RBF for crime rate prediction
2. **Regularization**: Strong regularization (α=100) improves generalization
3. **Feature Selection**: Group Lasso effectively identifies important features
4. **Model Robustness**: Linear models show consistent performance across different formulations
5. **Data Quality**: Proper preprocessing significantly impacts model performance

## Future Work

- **Feature Engineering**: Explore interaction terms and polynomial features
- **Ensemble Methods**: Combine multiple models for improved performance
- **Deep Learning**: Investigate neural networks for both regression and classification
- **Interpretability**: Analyze feature importance and model explanations
- **Cross-Domain Validation**: Test models on different geographic regions

---

*This project demonstrates comprehensive machine learning analysis covering both regression and classification problems, with emphasis on proper methodology, hyperparameter tuning, and result interpretation.* 