# credit_risk_app

# 💳 Credit Risk Analysis & AI Assistant

Ứng dụng web phân tích rủi ro tín dụng sử dụng Machine Learning kết hợp AI Chatbot để hỗ trợ giải thích kết quả mô hình một cách trực quan và dễ hiểu.

---

## 🚀 Overview

Project này cho phép:

* Phân tích dữ liệu tín dụng
* Huấn luyện và so sánh nhiều mô hình ML
* Hiển thị các chỉ số đánh giá
* Sử dụng AI để giải thích kết quả

---

## 📌 Features

* 📊 Upload dữ liệu tín dụng
* 🔄 Data preprocessing
* 🤖 Train & compare multiple models
* 📈 Hiển thị metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * KS
* 🧠 AI Chatbot:

  * Phân tích kết quả mô hình
  * So sánh các thuật toán
  * Giải thích bằng tiếng Việt

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas, NumPy
* Groq API (Llama 3 LLM)

---

## 📁 Project Structure

```
credit_risk_app/
│
├── app.py
├── requirements.txt
├── .env (not included in repo)
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
```

---

## ⚙️ Installation & Run

### 1. Clone repository

```bash
git clone https://github.com/buimanh86/credit_risk_app.git
cd credit_risk_app
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Tạo file `.env` trong thư mục gốc:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run application

```bash
streamlit run app.py
```

---

## 🔐 Security Note

* Không commit file `.env`
* Không hard-code API key trong source code
* Sử dụng environment variables để bảo mật

---

## 🤖 AI Chatbot

Chatbot sử dụng Groq API với model:

```
llama-3.3-70b-versatile
```

Chức năng:

* Phân tích kết quả mô hình
* So sánh hiệu suất giữa các model
* Hỗ trợ giải thích dữ liệu tín dụng

---

## 📈 Future Improvements

* Deploy lên Streamlit Cloud
* Thêm nhiều model hơn
* Tối ưu UI/UX
* Lưu lịch sử phân tích

---

## 👨‍💻 Author

* GitHub: https://github.com/buimanh86

---

## ⭐ Support

Nếu bạn thấy project hữu ích, hãy ⭐ repo để ủng hộ!
