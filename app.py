
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("UAE Customer Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Encode categorical
    df_encoded = df.copy()
    le_dict = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le

    # Classification
    st.subheader("Classification Model")
    X = df_encoded.drop("Will_Buy", axis=1)
    y = df_encoded["Will_Buy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # ROC Curve
    y_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)
    st.write(feat_df)

    # Clustering
    st.subheader("Customer Segmentation")
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(X)
    df["Cluster"] = clusters

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)

    fig2, ax2 = plt.subplots()
    ax2.scatter(comps[:,0], comps[:,1], c=clusters)
    st.pyplot(fig2)

    # Regression
    st.subheader("Spending Prediction")
    y_reg = df_encoded["Max_Spend"]
    X_reg = df_encoded.drop("Max_Spend", axis=1)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2)
    reg = RandomForestRegressor()
    reg.fit(Xr_train, yr_train)

    preds = reg.predict(Xr_test)
    st.write("Sample Predictions:", preds[:5])

    # Upload new data
    st.subheader("Upload New Customers for Prediction")
    new_file = st.file_uploader("Upload New Customer Data", key="new")

    if new_file:
        new_df = pd.read_csv(new_file)

        for col in new_df.columns:
            if col in le_dict:
                new_df[col] = le_dict[col].transform(new_df[col].astype(str))

        pred = clf.predict(new_df)
        st.write("Predictions:", pred)
