# Methodology

This document describes the methodological choices made while designing the fraud detection system, including data preprocessing, model selection, and evaluation strategy.

The guiding principle throughout the project was to align model complexity with dataset characteristics while maintaining interpretability and robustness.

---

## Data Preprocessing

The raw dataset contains transactional, temporal, and demographic features. Prior to model training, the following preprocessing steps were applied:

- **Removal of identifier-like columns**  
  Columns such as `Id` and `upi_number` were excluded to prevent data leakage and overfitting, as they do not carry predictive value.

- **Feature scaling**  
  Numerical features (e.g., transaction amount, temporal attributes, age) were standardized using feature scaling to ensure equal contribution during model training.

- **Categorical encoding**  
  Categorical attributes such as transaction category, state, and zip code were transformed using one-hot encoding to make them compatible with linear classification models.

- **Train–test split**  
  The dataset was split into training and test sets using stratified sampling to preserve the class distribution of fraudulent and non-fraudulent transactions.

---

## Handling Class Imbalance

Fraud detection datasets are inherently imbalanced, with fraudulent transactions forming a minority class.

Instead of using resampling techniques, **class weighting** was applied within the classification model. This approach:

- assigns higher penalty to misclassified fraud cases  
- preserves the original data distribution  
- avoids introducing synthetic data  
- simplifies evaluation and interpretation  

Class weighting provides a principled way to address imbalance while maintaining model stability.

---

## Model Selection

A **Logistic Regression** classifier was selected as the primary model for this project.

This choice was motivated by several factors:

- strong performance on binary classification problems
- robustness on small-to-medium-sized tabular datasets
- low risk of overfitting compared to complex models
- interpretability of learned coefficients
- reliable probability estimates for ROC-AUC computation

The model complexity was deliberately kept proportional to the dataset size to ensure generalization and reproducibility.

---

## Evaluation Strategy

Model performance was evaluated using multiple complementary metrics:

- **Accuracy** to measure overall correctness
- **F1-score** to balance precision and recall under class imbalance
- **ROC-AUC** to assess discriminative ability across thresholds
- **Cross-validated ROC-AUC** to evaluate stability across data splits

Each stage of the pipeline was tested independently before integration into the final end-to-end runner script, ensuring correctness at every step.

---

## Summary

The methodology emphasizes simplicity, interpretability, and robustness.  
By combining thoughtful preprocessing, an appropriate model choice, and comprehensive evaluation, the system effectively addresses the fraud detection task without unnecessary complexity.
