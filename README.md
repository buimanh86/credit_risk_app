# credit_risk_app

💳 Credit Risk Analysis & AI Assistant

Ứng dụng web phân tích rủi ro tín dụng sử dụng Machine Learning và tích hợp AI chatbot để hỗ trợ giải thích kết quả.

📌 Tính năng chính

📊 Upload dữ liệu tín dụng

🔄 Tiền xử lý dữ liệu

🤖 Huấn luyện và so sánh nhiều mô hình

📈 Hiển thị các metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, KS

🧠 AI Chatbot:

Giải thích kết quả mô hình

So sánh các model

Trả lời bằng tiếng Việt

🛠️ Công nghệ sử dụng

Python

Streamlit

Scikit-learn

Pandas / NumPy

Groq API (LLM - Llama 3)

📁 Cấu trúc project
credit_risk_app/
│
├── app.py
├── requirements.txt
├── .env (không push lên GitHub)
│
├── app_pages/
│   ├── chatbot.py
│   ├── model_comparison.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── risk_analysis.py
│   └── upload_data.py
│
├── utils/
│   ├── metrics.py
│   ├── model_utils.py
│   └── preprocessing_utils.py
⚙️ Cài đặt & chạy local

Clone repo:

git clone https://github.com/buimanh86/credit_risk_app.git
cd credit_risk_app

Tạo môi trường ảo:

python -m venv venv
venv\Scripts\activate

Cài thư viện:

pip install -r requirements.txt

Tạo file .env:

GROQ_API_KEY=your_api_key_here

Chạy ứng dụng:

streamlit run app.py
🔐 Lưu ý

Không commit file .env

Không hard-code API key trong code

Sử dụng biến môi trường để bảo mật

👨‍💻 Tác giả

GitHub: https://github.com/buimanh86
