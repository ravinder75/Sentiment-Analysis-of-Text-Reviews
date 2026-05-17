# Sentiment Analysis of Text Reviews

An end-to-end NLP Sentiment Analysis project that classifies IMDb movie reviews as Positive or Negative using Machine Learning, Deep Learning, and REST API deployment techniques.

---

# Project Overview

This project focuses on Natural Language Processing (NLP), Machine Learning, Deep Learning, and API deployment.

The application analyzes movie review text and predicts whether the sentiment is:
- Positive
- Negative

The project includes:
- Text preprocessing
- Feature engineering
- Machine Learning models
- Deep Learning (LSTM)
- REST API development using Flask
- Model deployment preparation

---

# Technologies Used

## Programming Language
- Python

## Data Science Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Machine Learning
- Scikit-learn
- Logistic Regression

## Deep Learning
- TensorFlow
- Keras
- LSTM Networks

## NLP
- NLTK
- TF-IDF Vectorization

## Backend Development
- Flask REST API

## Tools & Platforms
- Git
- GitHub
- Jupyter Notebook
- Linux
- VS Code

---

# Dataset

Dataset Used:
- IMDb Movie Reviews Dataset

Labels:
- Positive
- Negative

---

# Project Structure

```text
sentiment-analysis-project/
│
├── data/
│   └── IMDB Dataset.csv
│
├── notebooks/
│   └── 01-data-exploration-and-cleaning.ipynb
│
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── lstm_model.h5
│
├── api/
│   └── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# Step 1 — Setup & Data Collection ✅

Completed:
- Project directory setup
- Python virtual environment creation
- Jupyter Notebook setup
- Dataset loading using Pandas
- Dataset inspection
- Missing value analysis
- Initial exploration

---

# Step 2 — Text Preprocessing ✅

Completed:
- Text lowercasing
- HTML tag removal
- Punctuation removal
- Stopword removal
- Tokenization
- Lemmatization
- Text cleaning pipeline

Techniques Used:
- Regex
- NLTK Stopwords
- WordNet Lemmatizer

---

# Step 3 — Feature Engineering (Vectorization) ✅

Completed:
- Feature and target definition
- Train-test splitting
- TF-IDF vectorization
- Sparse matrix generation
- Feature transformation

Concepts Learned:
- TF-IDF
- Sparse Matrices
- Feature Extraction

---

# Step 4 — Baseline Model Training & Evaluation ✅

Completed:
- Logistic Regression implementation
- Model training
- Prediction generation
- Classification report analysis
- Confusion matrix visualization
- Accuracy evaluation
- Model saving using Joblib

Machine Learning Techniques:
- Logistic Regression
- Classification Metrics
- Model Persistence

---

# Step 5 — Advanced Deep Learning Model (LSTM) ✅

Completed:
- TensorFlow installation
- Keras Tokenizer implementation
- Sequence preprocessing
- Sequence padding
- Embedding layer creation
- LSTM neural network design
- Dense output layer implementation
- Model compilation
- Deep learning model training
- Training history visualization
- Baseline comparison with ML model

Deep Learning Concepts:
- Recurrent Neural Networks (RNN)
- LSTM Networks
- Word Embeddings
- Sequential Modeling
- Neural Network Training

Frameworks Used:
- TensorFlow
- Keras

---

# Step 6 — Build a Prediction API ✅

Completed:
- Flask API setup
- Flask application initialization
- Model loading inside Flask
- TF-IDF vectorizer loading
- Text preprocessing inside API
- Prediction endpoint creation
- JSON request parsing
- Prediction response generation
- API testing

API Features:
- REST API architecture
- JSON input/output
- Sentiment prediction endpoint
- Real-time prediction support

Backend Technologies:
- Flask
- REST API
- Joblib

---

# Current Project Status

Completed:
- NLP preprocessing pipeline
- TF-IDF feature engineering
- Logistic Regression model
- LSTM deep learning model
- Flask REST API integration
- Model persistence

The project now supports:
- Machine Learning predictions
- Deep Learning predictions
- API-based sentiment prediction

---

# Upcoming Steps

## Step 7 — Interactive UI
- Frontend development
- User interaction interface
- Real-time sentiment prediction UI

## Step 8 — Deployment & Documentation
- Docker deployment
- Cloud hosting
- Production deployment
- Full project documentation

---

# Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-project.git
```

Move into project directory:

```bash
cd sentiment-analysis-project
```

Create virtual environment:

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Run Flask API:

```bash
python app.py
```

---

# Machine Learning Workflow

```text
Raw Text Reviews
        ↓
Text Cleaning
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Prediction
```

---

# Deep Learning Workflow

```text
Raw Text Reviews
        ↓
Tokenizer
        ↓
Integer Sequences
        ↓
Padding
        ↓
Embedding Layer
        ↓
LSTM Layer
        ↓
Dense Layer
        ↓
Sentiment Prediction
```

---

# API Example

## Endpoint

```text
POST /predict
```

## Sample Request

```json
{
    "review": "This movie was amazing and inspiring"
}
```

## Sample Response

```json
{
    "sentiment": "Positive"
}
```

---

# Learning Outcomes

Through this project, I learned:
- NLP preprocessing pipelines
- TF-IDF vectorization
- Logistic Regression
- Deep Learning with LSTM
- TensorFlow & Keras
- REST API development
- Flask backend integration
- Model deployment workflow

---

# Future Improvements

- BERT / Transformer Models
- Streamlit Frontend
- Docker Containerization
- CI/CD Pipeline
- Cloud Deployment
- Real-time Web Interface
- GPU optimization

---

# Author

Ravinder K