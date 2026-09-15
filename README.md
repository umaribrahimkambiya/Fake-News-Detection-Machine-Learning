# 🛡️ FakeNews AI

**AI-powered Fake News Detection Web Application using Machine Learning and NLP.**

FakeNews AI analyzes news articles and predicts whether they are **FAKE** or **REAL** using a trained Machine Learning model.

## ✨ Features

* 🤖 Fake / Real news prediction
* 🧠 NLP text preprocessing
* 🔤 TF-IDF feature extraction
* 📊 Machine Learning classification
* 🌐 Flask web application
* 📱 Responsive modern UI
* 🎲 Random news article generator
* 🔌 JSON prediction API

### 🚀 Planned Features

* 📈 Prediction confidence
* 🔍 Explainable AI
* 🧠 Important-word highlighting
* 🗂️ Prediction history
* 🔎 Searchable history
* 📰 Article & source analysis
* 📊 Model performance dashboard
* 📉 Confusion matrix & evaluation metrics
* 👤 User accounts
* 💾 Database integration
* 📥 PDF/CSV reports
* 🔌 REST API
* 🌓 Theme customization
* 📱 Progressive Web App (PWA)

## 🛠️ Tech Stack

**Backend**

* Python
* Flask
* Flask-WTF

**Machine Learning**

* Scikit-learn
* NLTK
* TF-IDF
* Passive Aggressive Classifier

**Data**

* Pandas
* Joblib

**Frontend**

* HTML
* CSS
* JavaScript
* Responsive UI

## 🧠 How It Works

```text
News Article
     ↓
Text Preprocessing
     ↓
Stop-word Removal
     ↓
Stemming
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Model
     ↓
FAKE / REAL
```

## 📂 Project Structure

```text
FakeNews-AI/
│
├── app.py
├── prediction_model.py
├── forms.py
├── home.html
├── model2.pkl
├── tfidfvect2.pkl
├── random_dataset.csv
├── Fake News Prediction.ipynb
└── README.md
```

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## ⚠️ Disclaimer

This application provides **ML-based predictions**, not definitive fact-checking. Results should be independently verified using reliable sources.

## 🎯 Project Goal

The goal is to evolve this project from a basic Fake News classifier into a complete **Explainable News Intelligence Platform** combining:

**Machine Learning + NLP + Explainable AI + Analytics + Database + REST API**

---

⭐ If you find this project interesting, feel free to explore the code and give it a star.
