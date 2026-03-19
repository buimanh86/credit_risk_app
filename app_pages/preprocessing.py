import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run():
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
        return

    # Lấy dữ liệu từ session_state
    df = st.session_state["df"].copy()
    
    st.header("Original Dataset")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

    st.header("Data Preprocessing")
    
    # =========================
    # AUTO DETECT TARGET COLUMN
    # =========================
    possible_targets = [
        "loan_status",
        "default",
        "bad_loan",
        "credit_risk",
        "target",
        "label",
        "class"
    ]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.warning("Target column not detected automatically.")
        target_col = st.selectbox(
            "Select Target Column",
            df.columns
        )

    st.success(f"Target column: {target_col}")

    # =========================
    # DETECT COLUMN TYPES
    # =========================
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # =========================
    # HANDLE MISSING VALUES
    # =========================
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # =========================
    # ONE HOT ENCODING (fix bias)
    # =========================
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # =========================
    # SPLIT FEATURES / TARGET
    # =========================
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # =========================
    # CHECK MINORITY CLASS SIZE
    # =========================
    class_counts = y.value_counts()
    minority_count = class_counts.min()

    if minority_count < 6:
        st.error(
            "Dataset is not suitable.\n\n"
            "The minority class contains too few samples. "
            "Please provide at least 6 samples in each class."
        )
        st.stop()

    # encode target nếu là text
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)

    st.subheader("Processed Dataset")
    st.dataframe(X.head())

    # Lưu vào session_state
    st.session_state["X_raw"] = X
    st.session_state["y"] = y
    st.success("Preprocessing completed!")