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

