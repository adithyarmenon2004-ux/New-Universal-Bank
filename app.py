
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Universal Bank Loan Dashboard", layout="wide")

st.title("📊 Universal Bank Personal Loan Analytics Dashboard")

# Load data
df = pd.read_csv("UniversalBank.csv")

st.subheader("Dataset Overview")
st.write(df.head())

# Drop unnecessary columns
X = df.drop(columns=["ID", "ZIPCode", "Personal Loan"])
y = df["Personal Loan"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("📈 Model Performance")

fig_roc = plt.figure()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig_cm)

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
st.pyplot(fig_roc)

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
st.write(results_df)

st.markdown("### Insights")
st.write("Higher recall ensures we capture more potential loan customers.")
st.write("Precision ensures marketing spend is not wasted.")

# Upload test file
st.subheader("📂 Upload Test Data for Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)

    best_model = RandomForestClassifier()
    best_model.fit(X_train, y_train)

    preds = best_model.predict(test_data)
    test_data["Predicted Loan"] = preds

    st.write(test_data.head())

    csv = test_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
