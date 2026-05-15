# Sentiment Analysis of Text Reviews

An end-to-end NLP Sentiment Analysis project that classifies movie reviews as Positive or Negative using Machine Learning and Deep Learning techniques.

---

# Project Overview

This project focuses on Natural Language Processing (NLP) and Sentiment Analysis using the IMDb Movie Reviews dataset.

The project workflow includes:
- Data collection
- Text preprocessing
- Feature engineering
- Machine Learning models
- Deep Learning models
- API development
- Frontend UI
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
- Jupyter Notebook

---

# Dataset

Dataset:
- IMDb Movie Reviews Dataset

Classification Labels:
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
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# Step 1 — Setup & Data Collection ✅

Completed:
- Project directory setup
- Python virtual environment setup
- Jupyter Notebook setup
- Dataset loading using Pandas
- Data inspection with `.head()`
- Shape analysis with `.shape`
- Missing value analysis
- Initial dataset exploration

---

# Step 2 — Text Preprocessing ✅

Completed:
- Text lowercasing
- HTML tag removal
- Punctuation removal
- Stopword removal
- Tokenization
- Lemmatization
- Text cleaning pipeline creation
- Rejoining processed tokens into clean text

Techniques Used:
- Regular Expressions (Regex)
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
- Text Vectorization
- TF-IDF
- Sparse Matrices
- Training vs Testing Data

---

# Upcoming Steps

## Step 4 — Baseline Model Training
- Logistic Regression
- Model training
- Predictions
- Accuracy evaluation
- Confusion matrix
- Classification report

## Step 5 — Deep Learning
- LSTM Neural Networks
- Sequential modeling

## Step 6 — Backend API
- Flask / FastAPI

## Step 7 — Frontend UI
- Interactive web interface

## Step 8 — Deployment
- Docker
- Cloud deployment

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

# Learning Outcomes

Through this project, I am learning:
- NLP preprocessing pipelines
- Text vectorization techniques
- Machine Learning workflows
- Model evaluation methods
- Deep Learning fundamentals
- API development
- Full-stack ML deployment

---

# Future Improvements

- BERT / Transformers
- Streamlit frontend
- Docker containerization
- Cloud deployment
- CI/CD automation
- Real-time sentiment prediction

---

# Author

Ravinder K