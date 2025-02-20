# Credit Card Anomaly Detection

![Image_Alt](Credit Card Anomaly Detection.png)

# 📌Overview
This project focuses on identifying fraudulent credit card transactions using machine learning techniques. The goal is to help credit card companies detect fraud efficiently, preventing unauthorized charges to customers. The dataset used is from Kaggle, featuring transactions by European cardholders in September 2013, with a highly imbalanced class distribution (frauds account for only 0.172% of transactions).

# 📊Dataset
**Context**
Detecting fraudulent transactions is critical for credit card companies to protect customers from unauthorized charges.

**Content**
- Source: Transactions from European cardholders in September 2013.
- Size: 284,807 transactions over two days, with 492 frauds (0.172% of total).
**Features:**
- V1, V2, ..., V28: Anonymized numerical features resulting from PCA transformation (due to confidentiality, original features are not provided).
- Time: Seconds elapsed between each transaction and the first transaction.
- Amount: Transaction amount, useful for cost-sensitive learning.
- Class: Target variable (0 = Normal, 1 = Fraud).
- Imbalance: Highly skewed dataset with frauds being a rare occurrence.

# 🔥Inspiration
The objective is to build a model to identify fraudulent transactions. Due to the class imbalance, accuracy is measured using the Area Under the Precision-Recall Curve (AUPRC) rather than traditional confusion matrix accuracy.

# 🛠️Methodology
**Dependencies**
The project uses the following Python libraries:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
    from sklearn.metrics import *
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from xgboost import XGBClassifier
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

# 🔍Approach
**Data Preprocessing:**
- Load the dataset (assumed to be stored in Google Drive or locally).
- Scale features like Amount and Time using StandardScaler.
**Modeling:**
- Supervised Models: Logistic Regression, Decision Tree, Random Forest.
- Unsupervised Models: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.
**Evaluation:**
- Use AUPRC for performance evaluation due to class imbalance.
- Visualize results with confusion matrices (e.g., using seaborn heatmaps).

# 🚀Column Descriptions
- Time: Timestamp of the transaction (seconds from the first transaction).
- V1-V28: PCA-transformed features (anonymized for confidentiality).
- Amount: Transaction amount in the issuer's currency.
- Class: Binary label (0 = Normal, 1 = Fraud).

# 📊Results
**Observations**
- **Unsupervised Models:**
   - Isolation Forest: Detected 747 errors, 99.73% accuracy, ~28% fraud detection rate.
   - Local Outlier Factor (LOF): Detected 4600 errors, 98.38% accuracy, ~2% fraud detection rate.
   - One-Class SVM: Detected 8516 errors, 70.09% accuracy, 0% fraud detection rate.
   - Conclusion: Isolation Forest outperformed LOF and SVM in both accuracy and fraud detection.
- **Supervised Models:**
   - Logistic Regression: 99.89% accuracy.
   - Decision Tree: 99.92% accuracy.
   - Random Forest: 99.92% accuracy.
   - XGBoost: Comparable performance to Random Forest.
- **Insights:**
  - Isolation Forest is the most effective for this imbalanced dataset.
  - Accuracy can be improved by increasing sample size, using deep learning, or employing complex anomaly 
    detection models (at the cost of computational resources).

  # 🚀Future Work
- Increase sample size for better accuracy.
- Experiment with deep learning models (e.g., autoencoders).
- Optimize computational efficiency for complex models.
