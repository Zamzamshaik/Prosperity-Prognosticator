# 🚀 Prosperity Prognosticator: Startup Success Predictor

An AI-powered Machine Learning web application that predicts whether a startup will **Succeed or Fail** based on funding, milestones, relationships, and other key business factors.

---

## 📌 Project Overview

Prosperity Prognosticator is a full-stack ML project that uses a trained Random Forest model to analyze startup data and predict success probability.
It includes:

* Data Collection & Preprocessing Pipeline
* Feature Engineering
* Machine Learning Model Training
* Flask Backend API
* Simple Interactive Frontend UI

---

## 🧠 Machine Learning Model

* Algorithm: Random Forest Classifier
* Accuracy: ~79%
* Cross Validation Score: ~0.79
* Features Used: 26 engineered startup features

---

## 📂 Project Structure

```
prognosticator-machine-learning/
│
├── backend/
│   ├── app.py              # Flask backend API
│   ├── ml_model.py         # Model loading & prediction logic
│   └── main.py             # Optional entry file
│
├── models/
│   ├── model.pkl           # Trained ML model
│   └── feature_columns.pkl # Feature column order
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── train.py
│
├── data/                   # Raw & processed datasets
├── reports/                # Model reports
├── myenv/                  # Virtual environment (ignored in git)
└── README.md
```

---

## ⚙️ Installation & Setup (Step-by-Step)

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/prosperity-prognosticator.git
cd prosperi
```
