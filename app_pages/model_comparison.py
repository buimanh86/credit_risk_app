import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run():
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]

        st.header("Model Comparison")
        st.dataframe(results_df)

        fig, ax = plt.subplots(figsize=(8, 3))
        # Plot bar chart từ results_df
        results_df.set_index("Model").plot(
            kind="bar",
            ax=ax,
            width=0.6
        )

        ax.set_xticklabels(
            results_df["Model"].tolist(), 
            rotation=0
        )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="Metrics"
        )

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Please run Machine Learning Models first to see comparison.")