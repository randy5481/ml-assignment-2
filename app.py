import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Heart Disease Classification - ML Assignment 2")
st.write("BITS Pilani | M.Tech AIML")
st.markdown("---")


# sidebar stuff
st.sidebar.title("Settings")

# dataset upload - only test data as mentioned in assignment
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")


# model selection dropdown
model_option = st.sidebar.selectbox("Select Model", [
    "Compare All Models",
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

test_split = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
rand_state = 42


if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.write("Expected dataset: Heart Disease dataset from Kaggle")
    st.write("Target column should be the last column (HeartDisease)")
    st.stop()



# load data
df = pd.read_csv(uploaded_file)

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(df.head())

# let user pick target column
target = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)


def preprocess(df, target_col, ts):
    data = df.dropna().copy()
    
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # encode target if needed
    if y.dtype == 'object':
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=rand_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, list(X.columns)