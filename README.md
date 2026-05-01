# 🔥 Customer Churn Prediction (Deep Learning + MLOps)

## 📌 Project Overview

This project predicts whether a customer will churn using a Deep Learning model built with TensorFlow and deployed using FastAPI.

---

## 🎯 Objectives

* Build a neural network model for churn prediction
* Apply preprocessing and feature engineering
* Deploy model using FastAPI
* Implement production-ready pipeline

---

## 🧠 Model Details

* Model: Feedforward Neural Network
* Input Features: Customer attributes
* Output: Churn probability (0 or 1)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* FastAPI
* NumPy / Pandas

---

## 📁 Project Structure

```
customer-churn-ml/
│
├── data/
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│
├── api/
│   └── app.py
│
├── model.keras
├── columns.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Train model

```
python -m src.train
```

### 3️⃣ Run API

```
python -m uvicorn api.app:app --reload
```

### 4️⃣ Open Swagger

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Sample API Input

```json
{
  "tenure": 24,
  "MonthlyCharges": 85,
  "TotalCharges": 2000,
  "SeniorCitizen": 1,
  "gender_Male": 1,
  "Partner_Yes": 1,
  "Dependents_Yes": 0
}
```

---

## 📊 Sample Output

```json
{
  "probability": 0.72,
  "churn": 1
}
```

---

## 📈 Model Performance

* Accuracy: ~89%
* Loss: ~0.36

---

## 🔥 Features

* Automatic feature handling
* Scaler + preprocessing pipeline
* Real-time predictions via API
* Production-ready structure

---

## 🚀 Future Improvements

* Add frontend UI
* Deploy on cloud (AWS / Render)
* Add monitoring dashboard

---

## 👨‍💻 Author

Bharath Kumar Reddy
