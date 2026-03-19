import streamlit as st
import pandas as pd

def run():
    uploaded_file = st.file_uploader("Upload Credit Risk Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("Dataset Uploaded Successfully")
        st.dataframe(df.head())