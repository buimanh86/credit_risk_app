import streamlit as st

st.set_page_config(layout="wide", page_title="Credit Risk AI")

st.title("AI-based Credit Risk Evaluation Platform")

# NAVIGATION
menu = st.sidebar.radio(
    "Navigation",
    [
        "Upload Dataset",
        "Dataset & Preprocessing",
        "Machine Learning Models",
        "Model Comparison",
        "Risk Analysis",
        "AI Chatbot"
    ]
)

# Điều hướng file bằng cách import logic từ folder pages
if menu == "Upload Dataset":
    from app_pages import upload_data
    upload_data.run()
elif menu == "Dataset & Preprocessing":
    from app_pages import preprocessing
    preprocessing.run()
elif menu == "Machine Learning Models":
    from app_pages import models
    models.run()
elif menu == "Model Comparison":
    from app_pages import model_comparison
    model_comparison.run()
elif menu == "Risk Analysis":
    from app_pages import risk_analysis
    risk_analysis.run()
elif menu == "AI Chatbot":
    from app_pages import chatbot
    chatbot.run()