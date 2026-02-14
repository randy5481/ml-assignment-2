"""
model_training.py
─────────────────
BITS Pilani | M.Tech AIML | Machine Learning – Assignment 2
Training 6 classification models on the Heart Disease dataset
and printining the full evaluation metrics table.

Usage:
    python model_training.py

Dataset:
    heart.csv  (Heart Disease Dataset from Kaggle)
    https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# 1. Loading the Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("BITS Pilani – ML Assignment 2")
print("Heart Disease Classification")
print("=" * 60)

df = pd.read_csv("heart.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"\nTarget distribution:\n{df['HeartDisease'].value_counts()}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
