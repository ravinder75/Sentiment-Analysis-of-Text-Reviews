# Sentiment Analysis of Text Reviews

An end-to-end NLP Sentiment Analysis project that classifies movie reviews as Positive or Negative using Machine Learning and Deep Learning techniques.

---

# Project Overview

This project focuses on Natural Language Processing (NLP) and Sentiment Analysis using the IMDb Movie Reviews dataset.

The workflow includes:
- Data collection
- Text preprocessing
- Feature engineering
- Machine Learning model training
- Model evaluation
- Deep Learning
- API development
- Frontend integration
- Deployment

---

# Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Joblib
- Jupyter Notebook

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
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# Step 1 — Setup & Data Collection ✅

Completed:
- Project folder setup
- Python virtual environment creation
- Jupyter Notebook setup
- IMDb dataset loading using Pandas
- Dataset inspection using `.head()`
- Shape analysis using `.shape`
- Missing value analysis
- Initial data exploration

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
- Rejoining processed tokens into cleaned text

Techniques Used:
- Regex
- NLTK Stopwords
- WordNet Lemmatizer

---

# Step 3 — Feature Engineering (Vectorization) ✅

Completed:
- Defined Features (X) and Target (y)
- Imported `train_test_split`
- Split dataset into training and testing sets
- Imported `TfidfVectorizer`
- Configured TF-IDF vectorizer
- Vectorized training data using `fit_transform`
- Transformed testing data using `transform`
- Verified TF-IDF matrix dimensions

Concepts Learned:
- Feature Extraction
- TF-IDF Vectorization
- Sparse Matrices
- Training vs Testing datasets

---

# Step 4 — Baseline Model Training & Evaluation ✅

Completed:
- Imported Logistic Regression model
- Initialized Logistic Regression classifier
- Trained model on TF-IDF vectors
- Generated predictions on test data
- Imported evaluation metrics
- Evaluated model using classification report
- Created confusion matrix heatmap
- Saved trained ML model
- Saved fitted TF-IDF vectorizer
- Used Joblib for model persistence

Machine Learning Techniques:
- Logistic Regression
- Classification Metrics
- Accuracy Evaluation
- Confusion Matrix
- Model Persistence

Evaluation Metrics Used:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Current Project Status

Completed:
- Data collection
- NLP preprocessing
- TF-IDF feature engineering
- Logistic Regression model training
- Model evaluation
- Model saving

Project now has a working sentiment classification pipeline.

---

# Upcoming Steps

## Step 5 — Deep Learning (LSTM)
- LSTM Neural Networks
- Sequential text modeling
- Deep learning sentiment analysis

## Step 6 — Prediction API
- Flask / FastAPI backend
- REST API endpoints

## Step 7 — Interactive UI
- Frontend web application
- User review prediction interface

## Step 8 — Deployment
- Docker containerization
- Cloud deployment
- Production hosting

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
Prediction & Evaluation
```

---

# Learning Outcomes

Through this project, I am learning:
- NLP preprocessing pipelines
- Feature engineering techniques
- TF-IDF vectorization
- Machine Learning workflows
- Logistic Regression
- Model evaluation
- Model persistence using Joblib
- End-to-end ML project development

---

# Future Improvements

- BERT / Transformer models
- Streamlit frontend
- Docker deployment
- CI/CD pipeline
- Cloud hosting
- Real-time sentiment prediction API

---

# Author

Ravinder K