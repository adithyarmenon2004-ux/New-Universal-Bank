
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Universal Bank AI Dashboard", layout="wide")

st.title("💼 Universal Bank - Personal Loan Intelligence Dashboard")

# Load data
df = pd.read_csv("UniversalBank.csv")
df.columns = df.columns.str.strip()

target_col = [col for col in df.columns if "Personal Loan" in col][0]

st.header("📊 Descriptive Analytics")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="Income", color=target_col, barmode="overlay",
                       title="Income Distribution vs Loan Acceptance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher income customers show higher acceptance probability.")

with col2:
    fig = px.box(df, x=target_col, y="CCAvg", color=target_col,
                 title="Credit Card Spend vs Loan")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Customers spending more are more likely to take loans.")

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
st.caption("Strong correlations help identify key drivers like income & CCAvg.")

# Model prep
X = df.drop(columns=[target_col, "ID", "ZIPCode"], errors="ignore")
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []
st.header("🤖 Model Performance")

roc_fig = plt.figure()

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

    cm = confusion_matrix(y_test, y_pred)
    cm_perc = cm / cm.sum()

    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig_cm)

    st.write(f"Percentage Distribution for {name}:")
    st.write(cm_perc)

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
st.pyplot(roc_fig)

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
st.dataframe(results_df)

st.header("🎯 Upload Data for Prediction")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    test = pd.read_csv(file)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(test)
    test["Predicted Loan"] = preds

    st.write(test.head())
    st.download_button("Download Predictions", test.to_csv(index=False), "predictions.csv")
