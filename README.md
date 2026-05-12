# Sentiment Analysis of Text Reviews

An end-to-end NLP Sentiment Analysis project that classifies movie reviews as Positive or Negative using Machine Learning and Deep Learning techniques.

---

# Project Overview

This project focuses on Natural Language Processing (NLP) and Sentiment Analysis using the IMDb Movie Reviews dataset.

The project includes:
- Data cleaning
- Text preprocessing
- Feature engineering
- Machine Learning models
- Deep Learning models
- API development
- Frontend integration
- Deployment

---

# Project Goals

- Learn Natural Language Processing (NLP)
- Understand text preprocessing techniques
- Build sentiment classification models
- Train Machine Learning algorithms
- Explore Deep Learning for NLP
- Build a prediction API
- Create an interactive UI
- Deploy a production-ready ML application

---

# Technologies Used

## Programming Language
- Python

## Data Analysis
- Pandas
- NumPy

## Visualization
- Matplotlib
- Seaborn

## Machine Learning
- Scikit-learn

## NLP
- NLTK

## Development Environment
- Jupyter Notebook

---

# Dataset

Dataset Used:
- IMDb Movie Reviews Dataset

The dataset contains movie reviews labeled as:
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

# Completed Steps

## Step 1 — Setup & Data Collection
Completed:
- Project directory setup
- Python virtual environment creation
- Jupyter Notebook setup
- Dataset loading using Pandas
- Initial data exploration
- Missing value analysis
- DataFrame inspection

---

## Step 2 — Text Preprocessing (In Progress)

Completed:
- Text lowercasing
- Applying custom functions to DataFrame columns
- Removing HTML tags
- Removing punctuation
- Downloading NLTK stopwords
- Tokenization
- Stopword removal
- WordNet lemmatization setup

Current Work:
- Applying lemmatization to tokenized reviews
- Rejoining cleaned tokens into final processed text

---

# Upcoming Steps

## Feature Engineering
- TF-IDF Vectorization
- Text feature extraction

## Machine Learning Models
- Logistic Regression
- Model evaluation
- Accuracy analysis

## Deep Learning
- LSTM Model
- Sequential text modeling

## Backend Development
- Flask/FastAPI API

## Frontend
- Interactive web interface

## Deployment
- Docker
- Cloud deployment

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ravinder75/Sentiment-Analysis-of-Text-Reviews
```

Move into project directory:

```bash
cd sentiment-analysis-project
```

Create virtual environment:

```bash
python3 -m venv venv
```

Activate virtual environment:

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
- Text cleaning techniques
- Machine Learning workflows
- Deep Learning basics
- API development
- Full-stack ML deployment

---

# Author

Ravinder K

---

# Future Improvements

- Real-time sentiment prediction
- Transformer models (BERT)
- Streamlit frontend
- Docker deployment
- Cloud hosting
- CI/CD pipeline