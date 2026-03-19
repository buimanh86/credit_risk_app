import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Import hàm KS từ utils
from utils.metrics import ks_statistic

def run():
    if "X_raw" not in st.session_state:
        st.warning("Please run preprocessing first")
        st.stop()

    X = st.session_state["X_raw"]
    y = st.session_state["y"]

    # =========================
    # CLASS DISTRIBUTION
    # =========================
    st.subheader("Class Distribution")
    fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
    sns.countplot(x=y, ax=ax_dist)
    ax_dist.set_title("Class Distribution")
    ax_dist.set_xlabel("Class")
    ax_dist.set_ylabel("Count")
    st.pyplot(fig_dist)

    class_ratio = y.value_counts(normalize=True)
    st.write("Class Ratio")
    st.dataframe(
        class_ratio, 
        use_container_width=False,
        column_config={
            "widgets": st.column_config.Column(width="medium"),
            "proportion": st.column_config.NumberColumn(format="%.4f", width="small")
        }
    )

    imbalance = class_ratio.min() < 0.3
    if imbalance:
        st.warning("Dataset appears to be imbalanced")

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # HANDLE IMBALANCE (SMOTE)
    # =========================
    if imbalance:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        st.success("SMOTE applied to training data")

    # =========================
    # FEATURE SCALING (FIXED TO PRESERVE INDEX)
    # =========================
    scaler = StandardScaler()
    
    # Giữ lại index gốc của X_train và X_test trước khi scale
    train_idx = X_train.index
    test_idx = X_test.index

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gán lại index cũ vào DataFrame mới sau khi scale thay vì reset_index
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=train_idx)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=test_idx)

    # =========================
    # MODELS DEFINITION
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=42)
    }

    results = []
    trained_models = {}

    st.header("Machine Learning Model Results")

    # =========================
    # TRAIN & EVALUATE
    # =========================
    for name, model in models.items():
        st.subheader(name)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        roc = roc_auc_score(y_test, prob)
        ks = ks_statistic(y_test, prob)

        # Display Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", round(acc, 3))
        c2.metric("Precision", round(precision, 3))
        c3.metric("Recall", round(recall, 3))
        c4.metric("F1", round(f1, 3))
        c5.metric("ROC-AUC", round(roc, 3))
        c6.metric("KS", round(ks, 3))

        # Charts: Confusion Matrix & ROC
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, prob)

        chart1, chart2 = st.columns(2)

        with chart1:
            fig, ax = plt.subplots(figsize=(3, 1.8))
            labels = np.array([["TN", "FP"], ["FN", "TP"]])
            annot = np.empty_like(cm).astype(str)
            for i in range(2):
                for j in range(2):
                    annot[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

            sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False,
                        xticklabels=["0", "1"], yticklabels=["0", "1"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.invert_yaxis()
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with chart2:
            fig2, ax2 = plt.subplots(figsize=(3, 1.8))
            ax2.plot(fpr, tpr)
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_title("ROC Curve")
            st.pyplot(fig2)

        # =========================
        # SPECIFIC MODEL DETAILS
        # =========================
        if name == "Logistic Regression":
            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_[0]
            })
            coef_df["Odds Ratio"] = np.exp(coef_df["Coefficient"])
            st.subheader("Logistic Regression Coefficients")

            coef_df = coef_df.sort_values("Odds Ratio", ascending=False)

            st.dataframe(
                coef_df,
                use_container_width=False, 
                column_config={
                    "Feature": st.column_config.TextColumn("Feature", width="medium"),
                    "Coefficient": st.column_config.NumberColumn(width="small"),
                    "Odds Ratio": st.column_config.NumberColumn(width="small")
                }
            )

        if name == "Random Forest":
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.subheader("Random Forest Feature Importance")

            st.dataframe(
                importance_df,
                use_container_width=False,
                column_config={
                    "Feature": st.column_config.TextColumn("Feature", width="medium"),
                    "Importance": st.column_config.NumberColumn(width="small", format="%.4f")
                }
            )
            st.write("OOB Score:", round(model.oob_score_, 3))

        if name == "XGBoost":
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.subheader("XGBoost Feature Importance")
    
            st.data_editor(
                importance_df,
                column_config={
                    "Feature": st.column_config.TextColumn("Feature", width="medium"),
                    "Importance": st.column_config.NumberColumn("Importance", width="small", format="%.4f"),
                },
                hide_index=False,
                use_container_width=False 
            )

        # Save results
        results.append({
            "Model": name, "Accuracy": acc, "Precision": precision,
            "Recall": recall, "F1 Score": f1, "ROC_AUC": roc, "KS": ks
        })
        trained_models[name] = model

    # Lưu vào session_state
    st.session_state["results_df"] = pd.DataFrame(results)
    st.session_state["trained_models"] = trained_models
    st.session_state["X_test"] = X_test