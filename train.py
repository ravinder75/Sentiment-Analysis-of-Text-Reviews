import pandas as pd
import numpy as np
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Download necessary NLTK corpora
print("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('wordnet')

# 2. Load dataset
dataset_path = 'data/IMDB Dataset.csv'
print(f"Loading dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} rows.")

# 3. Define text preprocessing functions (exactly as in the notebook)
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation & special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text).strip()
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words_set]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Rejoin tokens into a single string
    return ' '.join(tokens)

print("Preprocessing reviews (this may take a minute)...")
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Preprocessing completed.")

# 4. Separate features and target
X = df['cleaned_review']
y = df['sentiment']

# 5. Split dataset (identical to notebook setup)
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature engineering (TF-IDF Vectorization)
print("Fitting TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 7. Model training (Logistic Regression)
print("Training Logistic Regression model...")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_tfidf, y_train)
print("Model training completed.")

# 8. Save models
print("Saving model and vectorizer...")
# Create directories if they do not exist
os.makedirs('notebooks', exist_ok=True)

# Save to the locations expected by app.py
joblib.dump(lr_model, 'notebooks/lr_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# For extra safety/convenience, save copies in both locations
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(tfidf_vectorizer, 'notebooks/tfidf_vectorizer.pkl')

print("All assets saved successfully!")
