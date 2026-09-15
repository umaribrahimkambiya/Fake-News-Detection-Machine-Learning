# 🛡️ FakeNews AI — Fake News Detection System

> An end-to-end Machine Learning and Flask web application for detecting whether a news article is likely **FAKE** or **REAL** using Natural Language Processing (NLP) and TF-IDF feature extraction.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Stemming-purple)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Status](https://img.shields.io/badge/Status-Active-success)](#roadmap)

---

## 📌 Overview

**FakeNews AI** is a Machine Learning-powered web application that analyzes the text of a news article and predicts whether it is **FAKE** or **REAL**.

The project combines:

* Natural Language Processing
* Text preprocessing
* Stop-word removal
* Porter stemming
* TF-IDF vectorization
* N-gram feature extraction
* Machine Learning classification
* Flask web development
* Flask-WTF forms
* Serialized ML models using Joblib

The goal of the project is not simply to produce a prediction, but to provide a foundation for a more complete **news intelligence and explainable AI platform**.

---

## ✨ Features

### Current Features

* 📰 Enter a news article for analysis
* 🤖 Machine Learning-based classification
* 🚨 FAKE / REAL prediction
* 🧹 NLP text preprocessing
* ✂️ Stop-word removal
* 🌱 Porter stemming
* 🔢 TF-IDF feature extraction
* 🔤 Unigram, bigram and trigram features
* 🎲 Random/sample news generation
* 🌐 Flask web interface
* 📱 Responsive modern UI
* 🔌 JSON prediction endpoint
* 💾 Pre-trained model and vectorizer loading

### Advanced Platform Roadmap

The interface and architecture are designed to be extended with:

* 📈 Prediction confidence scores
* 🔍 Explainable AI
* 🧠 Important-word/feature highlighting
* 🗂️ Prediction history
* 🔎 Searchable analysis history
* 📰 Article metadata analysis
* 🌐 News-source analysis
* 📊 Model performance dashboard
* 📉 Confusion matrix and evaluation metrics
* 👤 User accounts
* 💾 Database integration
* 📥 Exportable analysis reports
* 🔌 REST API
* 🌓 Theme customization
* 📱 Progressive Web App (PWA) support

> **Note:** Some advanced features are planned architecture/features and require additional Flask, database, authentication, API, and ML work. They are not claimed as fully implemented by the current prediction pipeline.

---

# 🧠 Machine Learning Pipeline

The project follows a traditional NLP classification pipeline:

```text
                  News Article
                       │
                       ▼
              Text Preprocessing
                       │
          ┌────────────┴────────────┐
          │                         │
      Lowercase              Remove symbols
          │                         │
          └────────────┬────────────┘
                       ▼
                Tokenization
                       │
                       ▼
              Stop-word Removal
                       │
                       ▼
                 Stemming
                       │
                       ▼
              TF-IDF Vectorizer
              5,000 Features
                       │
                       ▼
             Machine Learning Model
                       │
                       ▼
                FAKE / REAL
```

---

# 🔬 NLP Preprocessing

Before classification, article text is processed to create a normalized representation.

The preprocessing pipeline includes:

### 1. Character Cleaning

Non-alphabetic characters are removed.

```python
re.sub('[^a-zA-Z]', ' ', text)
```

### 2. Lowercasing

```python
text.lower()
```

### 3. Tokenization

The text is split into individual words.

### 4. Stop-word Removal

Common English words are removed using the NLTK English stop-word list.

### 5. Porter Stemming

Words are reduced to their stem using:

```python
PorterStemmer()
```

For example:

```text
connecting
connected
connection
```

can be reduced toward a common stem representation.

---

# 🔢 TF-IDF Feature Extraction

The application uses:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3)
)
```

This allows the model to consider:

* Unigrams
* Bigrams
* Trigrams

Example:

```text
"government announced new policy"
```

can produce features such as:

```text
government
announced
new
policy
government announced
announced new
new policy
government announced new
announced new policy
```

The resulting text representation is converted into numerical features that can be processed by the classifier.

---

# 🤖 Machine Learning Model

The current serialized classifier is a:

**Passive Aggressive Classifier**

The model was trained to distinguish between:

```text
0 → FAKE
1 → REAL
```

The Flask application loads the trained model from:

```text
model2.pkl
```

and the TF-IDF vectorizer from:

```text
tfidfvect2.pkl
```

The prediction flow is:

```python
review = preprocess()

text_vect = tfidfvect.transform([review]).toarray()

prediction = model.predict(text_vect)
```

The application then converts the numerical prediction into:

```text
FAKE
```

or:

```text
REAL
```

---

# 🏗️ Project Architecture

```text
┌──────────────────────────────┐
│          Browser             │
│      Responsive UI           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          Flask App            │
│           app.py              │
├──────────────────────────────┤
│ Routes                        │
│ • /                           │
│ • /predict/<original_text>   │
│ • /random                     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      PredictionModel          │
│     prediction_model.py       │
├──────────────────────────────┤
│ • Preprocessing               │
│ • Stop-word removal           │
│ • Stemming                    │
│ • Vectorization               │
│ • Prediction                  │
└──────────────┬───────────────┘
               │
          ┌────┴─────┐
          ▼          ▼
┌──────────────┐ ┌──────────────┐
│ model2.pkl   │ │tfidfvect2.pkl│
│ ML Model     │ │ TF-IDF Model │
└──────────────┘ └──────────────┘
```

---

# 📂 Project Structure

A typical project structure is:

```text
Fake-News-Detector/
│
├── app.py
├── prediction_model.py
├── forms.py
│
├── home.html
│
├── model2.pkl
├── tfidfvect2.pkl
├── random_dataset.csv
│
├── nltk_data/
│   └── corpora/
│       └── stopwords/
│
├── Fake News Prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# 🌐 Flask Application

The web application is built using Flask.

## Main Route

```text
/
```

This route handles:

* Displaying the interface
* Generating sample articles
* Receiving article text
* Running predictions
* Displaying prediction results

---

## Prediction API

The application also provides:

```text
/predict/<original_text>
```

which returns JSON containing the prediction information.

Example response:

```json
{
    "original": "Example news article...",
    "preprocessed": "exampl news articl...",
    "prediction": "REAL"
}
```

---

## Random Article Endpoint

The application provides:

```text
/random
```

which retrieves a random article from:

```text
random_dataset.csv
```

Example response:

```json
{
    "title": "Example title",
    "text": "Example article text...",
    "label": "1"
}
```

---

# 🖥️ User Interface

The frontend has been redesigned as a responsive AI dashboard rather than a basic form.

The interface includes:

* Responsive navigation
* News analysis workspace
* Article text editor
* Character counter
* Sample article functionality
* Prediction result card
* Feature overview
* Model dashboard concept
* Future feature roadmap
* Mobile-friendly layouts
* Desktop dashboard layout

The design is intended to provide a foundation for turning the project into a larger **AI news-analysis platform**.

---

# 📊 Future Model Dashboard

The planned analytics dashboard will provide real evaluation results such as:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

The intended visualization is:

```text
                 Predicted
              FAKE       REAL
Actual FAKE    TN         FP
Actual REAL    FN         TP
```

These metrics should be calculated from a held-out evaluation dataset rather than manually entered values.

---

# 🔍 Explainable AI — Planned

A future version can expose the features that contributed most strongly to the classifier's decision.

For example:

```text
Prediction: FAKE

Important features:

government      ██████████
breaking        ████████
shocking        ██████
exclusive       █████
reportedly      ████
```

The purpose is to move the application from:

> "The model says FAKE."

toward:

> "The model says FAKE, and these features contributed strongly to the decision."

This will make the system more useful for demonstrations, research and educational purposes.

---

# 📈 Prediction Confidence — Planned

A future version can expose the classifier's decision score.

For example:

```text
Prediction
FAKE

Model score
-1.73
```

If a calibrated probability is desired, the model architecture should be updated appropriately rather than presenting a raw decision score as a probability.

---

# 🗂️ Prediction History — Planned

A database-backed history system can store:

```text
Analysis ID
User ID
Article
Prediction
Model Score
Timestamp
Source
Title
Author
```

Users could then:

* View previous analyses
* Search analyses
* Filter FAKE/REAL results
* Review previous predictions
* Export reports

---

# 💾 Database Architecture — Planned

The project can be extended with SQLite for development and PostgreSQL for production.

Possible database structure:

```text
Users
 │
 ├── Predictions
 │      │
 │      ├── Article
 │      ├── Prediction
 │      ├── Score
 │      └── Timestamp
 │
 └── Reports
```

A Flask ORM such as SQLAlchemy can be introduced when persistent data becomes necessary.

---

# 👤 User Authentication — Planned

Future versions can support:

```text
Register
   ↓
Login
   ↓
Dashboard
   ↓
Analyze Article
   ↓
Save Prediction
   ↓
View History
```

Security considerations include:

* Password hashing
* Session management
* Authentication
* Authorization
* Input validation
* CSRF protection
* Secure secret management

---

# 📥 Exportable Reports — Planned

Users will eventually be able to export an analysis containing:

```text
Article
Prediction
Confidence / Score
Important Features
Article Metadata
Source Information
Timestamp
```

Potential export formats:

```text
PDF
CSV
JSON
```

---

# 🔌 REST API — Planned Expansion

The existing prediction endpoint provides the foundation for a more structured API.

A future version could expose:

```text
POST /api/v1/predict
GET  /api/v1/history
GET  /api/v1/history/<id>
GET  /api/v1/metrics
POST /api/v1/reports
```

Example request:

```json
{
    "text": "News article text goes here..."
}
```

Example response:

```json
{
    "prediction": "REAL",
    "score": 0.84
}
```

---

# 📱 Progressive Web App — Planned

The frontend can eventually become a Progressive Web App by adding:

```text
manifest.json
service-worker.js
icons/
offline.html
```

This would allow the application to support:

* Installable web application
* Mobile-first experience
* Offline interface shell
* App-like navigation
* Cached static resources

---

# 🛠️ Technologies Used

## Programming

* Python
* HTML
* CSS
* JavaScript

## Machine Learning

* scikit-learn
* Passive Aggressive Classifier
* TF-IDF
* NLP

## Natural Language Processing

* NLTK
* Porter Stemmer
* English stop-word corpus

## Web Development

* Flask
* Flask-WTF
* WTForms
* Jinja2

## Data Processing

* Pandas

## Model Persistence

* Joblib

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository.

---

## 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify NLTK resources

The application expects the English stop-word corpus to be available.

The project can use its local:

```text
nltk_data/
```

directory.

If necessary, download the resource:

```python
import nltk
nltk.download("stopwords")
```

---

## 5. Run the application

```bash
python app.py
```

The Flask development server should then be available at:

```text
http://127.0.0.1:5000/
```

Open the address in your browser.

---

# 🧪 Example Usage

1. Open the application.
2. Paste a news article into the analysis workspace.
3. Click **Predict**.
4. The article is preprocessed.
5. The text is transformed using the saved TF-IDF vectorizer.
6. The trained classifier processes the feature vector.
7. The application returns:

```text
FAKE
```

or:

```text
REAL
```

---

# ⚠️ Model Compatibility

The serialized model and vectorizer were originally created using an older version of scikit-learn.

Because scikit-learn serialization is version-sensitive, loading old `.pkl` files with a significantly newer version can generate compatibility warnings or errors.

For a production-quality project, model artifacts should ideally be:

1. Re-trained with the target environment's supported library versions, or
2. Loaded using a compatible dependency environment.

The application currently uses the repaired vectorizer artifact compatible with the project's current runtime.

---

# 📚 Dataset

The training workflow used the:

**Fake and Real News Dataset**

The dataset contains examples of real and fake news articles and was used to train the text-classification model.

The training workflow used:

```text
True.csv
Fake.csv
```

with labels:

```text
REAL → 1
FAKE → 0
```

The project should comply with the dataset's licensing and usage requirements when redistributed.

---

# 🧪 Model Training Workflow

The training notebook follows approximately this workflow:

```text
Load True News
       │
       ▼
Load Fake News
       │
       ▼
Assign Labels
       │
       ▼
Combine Dataset
       │
       ▼
Shuffle
       │
       ▼
Remove Missing Values
       │
       ▼
NLP Preprocessing
       │
       ▼
TF-IDF
       │
       ▼
Train Classifier
       │
       ▼
Evaluate
       │
       ▼
Save Model
```

---

# 🔐 Security Considerations

Before deploying this application publicly, several security improvements should be implemented.

### Secret Key

Do not hard-code production secrets.

Instead of:

```python
app.config['SECRET_KEY'] = '...'
```

use an environment variable:

```python
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
```

### Input Validation

User-submitted article text should be validated and sanitized appropriately.

### API Security

A public API should include:

* Rate limiting
* Authentication where required
* Request size limits
* Error handling
* Logging
* Abuse protection

### Database Security

Production database credentials should never be committed to GitHub.

Use environment variables or a secure secret-management solution.

---

# 📌 Current Limitations

This project is a Machine Learning classification system, not a definitive fact-checking service.

A prediction of:

```text
FAKE
```

does **not automatically prove that an article is false**.

Likewise:

```text
REAL
```

does not guarantee that every claim in an article is factually correct.

The model identifies patterns learned from its training data.

Factors such as:

* New events
* Writing style
* Unseen topics
* Dataset bias
* Distribution changes
* Deliberately misleading content

can affect predictions.

Therefore, predictions should be treated as **model-assisted signals**, not absolute truth.

---

# 🗺️ Development Roadmap

## Phase 1 — Core ML Application

* [x] NLP preprocessing
* [x] TF-IDF vectorization
* [x] ML classifier
* [x] Flask application
* [x] Prediction interface
* [x] Random article generation
* [x] JSON prediction endpoint
* [x] Responsive frontend

## Phase 2 — Explainable AI

* [ ] Prediction score
* [ ] Confidence visualization
* [ ] Important feature extraction
* [ ] Important-word highlighting
* [ ] Explanation panel

## Phase 3 — Data & History

* [ ] SQLite/SQLAlchemy
* [ ] Prediction history
* [ ] Search and filtering
* [ ] Article metadata
* [ ] Source analysis

## Phase 4 — Analytics

* [ ] Model dashboard
* [ ] Accuracy
* [ ] Precision
* [ ] Recall
* [ ] F1 score
* [ ] Confusion matrix
* [ ] Evaluation charts

## Phase 5 — User Platform

* [ ] User registration
* [ ] Login/logout
* [ ] User dashboard
* [ ] Personal prediction history
* [ ] Saved reports

## Phase 6 — API & Deployment

* [ ] Versioned REST API
* [ ] API authentication
* [ ] Rate limiting
* [ ] Exportable reports
* [ ] PWA support
* [ ] Production deployment
* [ ] Monitoring and logging

---

# 🎯 Project Goals

The long-term goal is to evolve FakeNews AI from a simple binary classifier into a broader **AI-powered news analysis platform**.

The envisioned system will combine:

```text
Machine Learning
       +
Natural Language Processing
       +
Explainable AI
       +
News Metadata
       +
Source Analysis
       +
Prediction History
       +
Analytics
       +
REST API
       +
User Accounts
```

This architecture provides a foundation for demonstrating skills across:

* Machine Learning
* NLP
* Python
* Flask
* Frontend development
* Database design
* API development
* Explainable AI
* Data visualization
* Software architecture

---

# 👨‍💻 Portfolio Project

This project demonstrates the integration of a trained Machine Learning model into a real web application.

Rather than keeping the model inside a notebook, the project exposes the trained model through a Flask application and provides a user-facing interface for real-time text analysis.

### Key areas demonstrated

```text
Python
├── Machine Learning
├── NLP
├── Data Processing
├── Model Serialization
└── Backend Development

Flask
├── Routing
├── Forms
├── Templates
└── API Endpoints

Frontend
├── Responsive UI
├── JavaScript
├── Dashboard Design
└── Mobile Support
```

---

# 📜 Disclaimer

This project is intended for **educational, research and portfolio purposes**.

The predictions generated by the model should not be treated as authoritative fact-checking results.

Always verify important information using multiple reliable sources.

---

# ⭐ Future Vision

> **From Fake News Detection → to Explainable News Intelligence.**

The next stage of the project is to connect the responsive interface with real:

**confidence scoring → explainability → feature highlighting → database history → source analysis → analytics → user accounts → reports → REST API → PWA**

while keeping the existing Machine Learning pipeline stable and testable.

---

## 📄 License

Add the license appropriate for your project before publishing the repository, such as MIT, Apache-2.0, or another license compatible with the dataset and dependencies.
