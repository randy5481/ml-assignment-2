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

target = "HeartDisease"

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


def get_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # auc calculation
    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, probs), 4)
    except:
        auc = "N/A"

    metrics = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": auc,
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4)
    }
    return metrics, preds


# all 6 models
def get_all_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rand_state),
        "Decision Tree": DecisionTreeClassifier(random_state=rand_state),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=rand_state),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=rand_state, verbosity=0)
    }
    return models


def show_confusion_matrix(y_test, preds, name):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


if st.button("Run"):
    X_train, X_test, y_train, y_test, feat_names = preprocess(df, target, test_split)
    all_models = get_all_models()

    if model_option == "Compare All Models":
        st.markdown("---")
        st.subheader("All Models - Evaluation Metrics")

        results = {}
        all_preds = {}

        for name, mdl in all_models.items():
            m, p = get_metrics(mdl, X_train, X_test, y_train, y_test)
            results[name] = m
            all_preds[name] = p

        # show metrics table
        res_df = pd.DataFrame(results).T.reset_index()
        res_df.columns = ["Model"] + list(res_df.columns[1:])
        st.dataframe(res_df, use_container_width=True)

        # bar chart
        st.subheader("Accuracy vs F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = list(results.keys())
        acc_vals = [results[k]["Accuracy"] for k in x]
        f1_vals = [results[k]["F1 Score"] for k in x]
        x_pos = np.arange(len(x))
        bars1 = ax.bar(x_pos - 0.2, acc_vals, 0.4, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x_pos + 0.2, f1_vals, 0.4, label='F1 Score', color='coral')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, rotation=25, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("Model Comparison")
        plt.tight_layout()
        st.pyplot(fig)

        # confusion matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, (name, pred) in enumerate(all_preds.items()):
            with cols[i % 3]:
                fig_cm = show_confusion_matrix(y_test, pred, name)
                st.pyplot(fig_cm)

    else:
        # single model
        mdl = all_models[model_option]
        metrics, preds = get_metrics(mdl, X_train, X_test, y_train, y_test)

        st.markdown("---")
        st.subheader(f"Results - {model_option}")

        # show metrics
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("Accuracy", metrics["Accuracy"])
        c2.metric("AUC Score", metrics["AUC"])
        c3.metric("Precision", metrics["Precision"])
        c4.metric("Recall", metrics["Recall"])
        c5.metric("F1 Score", metrics["F1 Score"])
        c6.metric("MCC", metrics["MCC"])

        # confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = show_confusion_matrix(y_test, preds, model_option)
        st.pyplot(fig_cm)

        # classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, preds)
        st.text(report)

        # feature importance for tree models
        if model_option in ["Decision Tree", "Random Forest", "XGBoost"]:
            st.subheader("Feature Importance")
            imp = mdl.feature_importances_
            fi = pd.DataFrame({"Feature": feat_names, "Importance": imp})
            fi = fi.sort_values("Importance", ascending=False)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            sns.barplot(data=fi, x="Importance", y="Feature", palette="coolwarm", ax=ax2)
            ax2.set_title("Feature Importances")
            plt.tight_layout()
            st.pyplot(fig2)

st.markdown("---")
st.write("ML Assignment 2 | Heart Disease Dataset | 6 Models")