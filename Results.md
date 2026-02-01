# Results

This document summarizes the performance of the fraud detection model on the UPI transaction dataset.

All reported metrics were obtained by executing the complete end-to-end pipeline using the final trained model.

---

## Test Set Performance

The Logistic Regression model achieved the following results on the held-out test set:

- **Accuracy**: 0.8801  
- **F1-score**: 0.8947  
- **ROC-AUC**: 0.9543  

These results indicate that the model is able to effectively distinguish between fraudulent and legitimate transactions.

---

## Metric Interpretation

- **Accuracy** reflects the overall correctness of predictions but can be misleading in imbalanced datasets.
- **F1-score** balances precision and recall, making it particularly relevant for fraud detection where false negatives are costly.
- **ROC-AUC** measures the model’s ability to rank fraudulent transactions higher than legitimate ones across all classification thresholds.

The high ROC-AUC score demonstrates strong discriminative capability.

---

## Cross-Validation Performance

In addition to test set evaluation, the model was evaluated using k-fold cross-validation with ROC-AUC as the scoring metric.

Cross-validation confirmed that model performance is **stable across different data splits**, indicating good generalization and robustness.

---

## Summary of Results

The results demonstrate that a well-preprocessed dataset combined with an interpretable baseline classification model can achieve strong performance on fraud detection tasks.

The evaluation confirms that the implemented pipeline is reliable, reproducible, and suitable for practical fraud detection scenarios.
