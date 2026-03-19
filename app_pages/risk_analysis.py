import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run():
    if "results_df" in st.session_state and "trained_models" in st.session_state:
        results_df = st.session_state["results_df"]
        trained_models = st.session_state["trained_models"]
        X_test = st.session_state["X_test"]

        st.header("AI Generated Insights")

        # Tìm model tốt nhất dựa trên ROC_AUC
        best_model_info = results_df.loc[results_df["ROC_AUC"].idxmax()]

        st.write(f"Best Model: **{best_model_info['Model']}**")
        st.write(f"ROC-AUC: **{best_model_info['ROC_AUC']:.3f}**")
        st.write(f"Accuracy: **{best_model_info['Accuracy']:.3f}**")
        st.write(f"Recall: **{best_model_info['Recall']:.3f}**")
        st.write(f"Precision: **{best_model_info['Precision']:.3f}**")

        st.markdown("---")

        st.header("Credit Risk Assessment")

        best_model_name = best_model_info["Model"]
        best_model = trained_models[best_model_name]

        min_idx = 0
        max_idx = len(X_test) - 1

        col1, col2 = st.columns([3, 1])

        with col1:
            user_index = st.number_input(
                f"Select Customer Index (from {min_idx} to {max_idx})",
                min_value=min_idx,
                max_value=max_idx,
                step=1
            )

        with col2:
            apply_button = st.button("Apply")

        if apply_button:
            sample = X_test.iloc[[int(user_index)]]
            prob = best_model.predict_proba(sample)[0][1]
            pred = best_model.predict(sample)[0]

            st.write("Probability of Default:", round(prob, 3))
            st.write("Predicted Class:", pred)

            if prob < 0.3:
                risk_level = "Low"
                st.success("Low Risk Borrower")
            elif prob < 0.6:
                risk_level = "Medium"
                st.warning("Medium Risk Borrower")
            else:
                risk_level = "High"
                st.error("High Risk Borrower")
            
            st.write("Risk Level:", risk_level)

        # Risk Browser
        st.header("Risk Prediction Browser")
        probs = best_model.predict_proba(X_test)[:, 1]
        risk_df = X_test.copy()
        risk_df["Default Probability"] = probs
        risk_df["Risk Level"] = pd.cut(
            probs,
            bins=[0, 0.3, 0.6, 1],
            labels=["Low", "Medium", "High"]
        )

        risk_filter = st.selectbox(
            "Filter Risk Level",
            ["All", "Low", "Medium", "High"]
        )

        if risk_filter != "All":
            filtered_df = risk_df[risk_df["Risk Level"] == risk_filter]
        else:
            filtered_df = risk_df

        st.dataframe(filtered_df)

        # Risk Distribution Chart
        st.header("Risk Distribution")
        fig, ax = plt.subplots(figsize=(4, 2))
        risk_df["Risk Level"].value_counts().reindex(["Low", "Medium", "High"]).plot(
            kind="bar",
            ax=ax,
            width=0.6,
            color="#4C72B0"
        )
        ax.set_title("Risk Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Run Machine Learning Models first")