# WiDS5.0
Anomaly Detection using Probabilistic ML
# WiDS5.0 – UPI Fraud Detection System

This repository implements an end-to-end machine learning pipeline for detecting fraudulent UPI transactions using a real-world dataset.

The project focuses on building a **clear, modular, and reproducible fraud detection workflow**, covering all essential stages of a machine learning system—from data loading and preprocessing to model training and evaluation.

---

## 🎯 Project Objective

The objective of this project is to design and evaluate a reliable fraud detection model that can distinguish between legitimate and fraudulent UPI transactions using transaction-level data.

Key goals include:
- understanding fraud-related data characteristics
- handling class imbalance effectively
- selecting an appropriate classification model
- evaluating performance using standard machine learning metrics

---

## 📊 Dataset Description

The dataset consists of **2666 UPI transactions** with the following feature groups:

- **Temporal features**: transaction hour, day, month, and year  
- **Demographic features**: customer age and state  
- **Transaction attributes**: transaction amount, category, and zip code  
- **Target variable**: `fraud_risk` (binary classification)

Fraud cases represent a smaller fraction of the dataset, reflecting the inherent class imbalance found in real-world fraud detection problems.

---

## 🧱 Project Structure

The repository is organized in a modular manner, with each component of the machine learning pipeline placed in a dedicated script. This design improves readability, testability, and reproducibility.

