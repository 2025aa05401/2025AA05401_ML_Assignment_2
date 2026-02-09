import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("💼 Adult Income Classification – Streamlit App")

st.write("""
Upload **test data only** (CSV format) from the Adult Income dataset  
and evaluate different classification models.
""")

# -------------------------------
# (a) Dataset Upload Option
# -------------------------------
st.header("📂 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # -------------------------------
    # Separate Features & Target
    # -------------------------------
    if "income" not in data.columns:
        st.error("Target column 'income' not found in dataset!")
        st.stop()

    X = data.drop("income", axis=1)
    y = data["income"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # (b) Model Selection Dropdown
    # -------------------------------
    st.header("🤖 Select Classification Model")

    model_name = st.selectbox(
        "Choose a model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(eval_metric="logloss", random_state=42)

    # -------------------------------
    # Train Model
    # -------------------------------
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # -------------------------------
    # (c) Display Evaluation Metrics
    # -------------------------------
    st.header("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col1.metric("AUC", round(roc_auc_score(y, y_prob), 4))

    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col2.metric("Recall", round(recall_score(y, y_pred), 4))

    col3.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    # -------------------------------
    # (d) Confusion Matrix
    # -------------------------------
    st.header("🧩 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"]
    )

    st.dataframe(cm_df)

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.header("📄 Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(4))

else:
    st.info("Please upload a CSV file to continue.")
