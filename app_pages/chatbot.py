import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
def run():
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    st.header("AI Credit Risk Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Lấy kết quả model từ session_state
        model_results = ""
        if "results_df" in st.session_state:
            model_results = st.session_state["results_df"].to_string(index=False)

        system_prompt = f"""
        Bạn là trợ lý AI phân tích kết quả mô hình đánh giá rủi rủi ro tín dụng.
        Đây là bảng kết quả các mô hình:
        {model_results}
        Nhiệm vụ của bạn:
        - Phân tích và so sánh các mô hình
        - Giải thích Accuracy, Precision, Recall, F1, ROC-AUC, KS
        - Trả lời dựa trên bảng kết quả trên
        Trả lời bằng tiếng Việt, ngắn gọn và đúng trọng tâm.
        """

        history = st.session_state.messages[-6:]

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history
                ],
                temperature=0.2,
                max_tokens=600
            )

            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Lỗi kết nối Chatbot: {e}")