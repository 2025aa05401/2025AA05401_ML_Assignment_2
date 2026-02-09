import streamlit as st
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("💼 Adult Income Classification – Model Comparison")
st.write("""
This application compares multiple machine learning classification models  
used to predict whether an individual's income exceeds **$50K per year**.
""")

# -------------------------------
# Dataset Description
# -------------------------------
st.header("📊 Dataset Description")

st.markdown("""
- **Dataset:** Adult Income Dataset (UCI Machine Learning Repository)
- **Instances:** 48,842
- **Features:** 14
- **Target Variable:**
  - `<=50K` → Low Income  
  - `>50K` → High Income
- **Problem Type:** Binary Classification
""")

# -------------------------------
# Evaluation Metrics Table
# -------------------------------
st.header("📈 Model Evaluation Metrics")

results_data = {
    "ML Model Name": [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ],
    "Accuracy": [0.827883, 0.812529, 0.834024, 0.808076, 0.859205, 0.876248],
    "AUC": [0.860793, 0.753214, 0.856943, 0.864383, 0.910814, 0.928644],
    "Precision": [0.724623, 0.604707, 0.671117, 0.704370, 0.740576, 0.776087],
    "Recall": [0.459821, 0.639031, 0.609056, 0.349490, 0.639031, 0.683036],
    "F1 Score": [0.562622, 0.621395, 0.638582, 0.467178, 0.686066, 0.726594],
    "MCC": [0.480593, 0.497277, 0.532241, 0.399403, 0.598644, 0.649242]
}

results_df = pd.DataFrame(results_data)

st.dataframe(results_df, use_container_width=True)

# -------------------------------
# Observations Section
# -------------------------------
st.header("🧠 Model Performance Observations")

observations = {
    "Logistic Regression":
        "Good accuracy and AUC but low recall, indicating difficulty in identifying high-income individuals.",
    "Decision Tree":
        "Balanced precision and recall but lower AUC, suggesting overfitting.",
    "kNN":
        "Improved F1 score and MCC; performance depends on feature scaling and choice of k.",
    "Naive Bayes":
        "High AUC but very low recall due to independence assumptions and class imbalance.",
    "Random Forest (Ensemble)":
        "Strong performance across metrics by capturing non-linear patterns and reducing overfitting.",
    "XGBoost (Ensemble)":
        "Best overall performer; effectively handles class imbalance and complex feature interactions."
}

for model, obs in observations.items():
    st.subheader(model)
    st.write(obs)

# -------------------------------
# Conclusion
# -------------------------------
st.header("✅ Conclusion")

st.markdown("""
- **Ensemble models outperform individual classifiers**
- **XGBoost** achieved the highest Accuracy, AUC, F1 Score, and MCC
- **AUC and MCC** are more reliable than accuracy for imbalanced datasets like Adult Income
""")

st.success("🎯 XGBoost is the most suitable model for this dataset.")
