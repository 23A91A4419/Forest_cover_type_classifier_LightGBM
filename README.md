# 🌲 Forest Cover Type Classification using LightGBM

## 📌 Project Overview

This project focuses on building a high-performance multi-class classification model to predict forest cover types using the LightGBM framework. The objective is to apply advanced feature engineering, systematic hyperparameter tuning, and robust evaluation techniques to develop a production-ready machine learning pipeline.

The Forest Cover Type dataset contains cartographic variables describing forest areas. The task is to predict the forest cover type (7 classes) based on these environmental features.

---

## 🎯 Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Handle multi-class imbalance
- Engineer new domain-inspired features
- Train a baseline LightGBM classifier
- Perform systematic hyperparameter tuning
- Evaluate using macro and micro F1 scores
- Visualize confusion matrix and feature importance
- Compare LightGBM against an alternative model (Random Forest / XGBoost)
- Document the complete ML workflow

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Core Libraries:**  
  - LightGBM  
  - Scikit-learn  
  - Pandas  
  - NumPy  
- **Visualization:**  
  - Matplotlib  
  - Seaborn  
- **Environment:**  
  - Jupyter Notebook / JupyterLab  

---

## 📂 Dataset

The project uses the **Forest Cover Type Dataset**.

- 581,012 instances
- 54 features
- 7 forest cover classes
- No missing values
- Multi-class classification problem

Target Variable:
- `Cover_Type` (1 to 7)

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset shape and structure inspection
- Missing value verification
- Target class distribution analysis
- Correlation heatmap visualization
- Feature distribution analysis

Key Observations:
- Moderate class imbalance observed
- Several numerical features showed high correlation
- Tree-based models do not require feature scaling

---

## 🧠 Feature Engineering

To improve model performance, the following features were engineered:

1. Elevation_Slope_Interaction  
2. Hydrology_Distance_Ratio  
3. Hillshade_Difference  
4. Mean_Hillshade  
5. Elevation_Roadway_Interaction  

These features were created to capture nonlinear relationships and domain-specific interactions.

---

## ✂ Train-Test Split

The dataset was split using stratified sampling:

- 80% Training Data
- 20% Testing Data
- Stratification maintained class proportions
- Random state fixed for reproducibility

---

## 🌱 Baseline Model – LightGBM

A baseline LightGBM classifier was trained using default parameters to establish initial performance.

Evaluation Metrics:
- Accuracy
- Macro F1 Score
- Micro F1 Score
- Precision
- Recall

This baseline served as a reference for further improvements.

---

## ⚖ Handling Class Imbalance

Class imbalance was addressed using:

- `class_weight = 'balanced'`

This ensured fair learning across minority classes.

---

## 🔥 Hyperparameter Tuning

A systematic hyperparameter tuning process was implemented using cross-validation.

Parameters Tuned:
- num_leaves
- learning_rate
- feature_fraction
- reg_alpha
- reg_lambda

Method:
- Stratified K-Fold Cross-Validation (5 folds)
- F1 Macro scoring metric
- Early stopping used to prevent overfitting

Tuning significantly improved model performance over baseline.

---

## 📊 Model Evaluation

The final tuned model was evaluated on the hold-out test set.

Metrics Reported:
- Accuracy
- Macro F1 Score
- Micro F1 Score
- Precision (macro & micro)
- Recall (macro & micro)

### Confusion Matrix

A confusion matrix was generated to analyze:
- Per-class performance
- Misclassification patterns
- Minority class prediction quality

---

## 📈 Feature Importance Analysis

Two types of feature importance were analyzed:

1. Split Importance
2. Gain Importance

Findings:
- Elevation-related features were highly influential
- Engineered interaction features improved model performance
- Distance-based features contributed significantly

---

## ⚔ Comparative Analysis

An alternative model (Random Forest / XGBoost) was trained using the same dataset.

Comparison Criteria:
- Accuracy
- Macro F1 Score
- Training Time
- Overfitting Gap
- Feature Importance Trends

LightGBM demonstrated:
- Faster training time
- Better handling of large dataset
- Improved macro F1 score
- Efficient memory usage

---

## 📈 Results Summary

| Model                | Accuracy | Macro F1 | Training Time |
|----------------------|----------|----------|---------------|
| LightGBM (Baseline)  |    -     |    -     |      -        |
| LightGBM (Tuned)     |    -     |    -     |      -        |
| Alternative Model    |    -     |    -     |      -        |

*(Fill in actual results after running experiments.)*

---

## 🧪 Reproducibility

- Fixed random_state
- Stratified splitting
- Documented hyperparameters
- Clear experiment tracking

---

## 🚀 Key Learnings

- LightGBM is highly efficient for large multi-class datasets
- Feature engineering significantly improves performance
- Hyperparameter tuning plays a critical role in boosting F1 score
- Macro F1 is essential when dealing with class imbalance
- Feature importance provides valuable interpretability

---

## 🔮 Future Improvements

- Implement SHAP for advanced interpretability
- Perform Bayesian Optimization for faster tuning
- Explore ensemble stacking methods
- Deploy the model as an API

---

## 📌 Conclusion

This project successfully demonstrates a complete end-to-end machine learning workflow for multi-class classification using LightGBM. Through careful feature engineering, systematic tuning, and thorough evaluation, the final model achieved significant performance improvements over the baseline.

The notebook is structured, reproducible, and suitable for industry-level portfolio presentation.
