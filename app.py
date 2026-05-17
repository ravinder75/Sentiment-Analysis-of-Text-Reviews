# ==========================================
# Step 2: Create the Flask Application Entry Point
# (This app.py file serves as our entry point)
# ==========================================

# ==========================================
# Step 3: Import Essential Libraries for the Flask API
# ==========================================
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# Step 4: Load a Trained ML Model and Vectorizer in a Flask App
# ==========================================
# We load them in the global scope so they are loaded only once at application startup.
model = None
vectorizer = None

try:
    # First search locally, then search notebooks folder as a fallback
    if os.path.exists('lr_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model = joblib.load('lr_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    elif os.path.exists('notebooks/lr_model.pkl') and os.path.exists('notebooks/tfidf_vectorizer.pkl'):
        model = joblib.load('notebooks/lr_model.pkl')
        vectorizer = joblib.load('notebooks/tfidf_vectorizer.pkl')
    else:
        # Standard fallback to trigger expected FileNotFoundError if files are completely missing
        model = joblib.load('lr_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Make sure 'lr_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as app.py.")
except Exception as e:
    print(f"An error occurred while loading the model/vectorizer: {e}")

# ==========================================
# Step 5: Implement a Text Preprocessing Function in a Flask App
# ==========================================
# Ensure NLTK resources are available locally
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize NLP resources
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans and preprocesses raw review text exactly as done during model training.
    """
    # 1. Lowercase text
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 3. Remove punctuation & special characters, strip extra spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text).strip()
    # 4. Tokenize and remove NLTK stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words_set]
    # 5. Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # 6. Rejoin tokens into a single clean string
    return ' '.join(tokens)

# ==========================================
# Step 6: Initialize the Flask Application
# ==========================================
app = Flask(__name__)

# ==========================================
# Optional/UI Route: Home page serving a premium web interface
# ==========================================
@app.route('/')
def home():
    """
    Renders the beautiful, dark-themed sentiment analyzer UI.
    """
    return render_template('index.html')

# ==========================================
# Step 7: Define a Prediction Endpoint in Flask
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Server error: Model or vectorizer assets are missing.'}), 500

    try:
        # ==========================================
        # Step 8: Receive and Parse JSON Data in a Flask API Endpoint
        # ==========================================
        data = request.get_json(force=True)

        # ==========================================
        # Step 9: Implement Robust Input Validation in a Flask API
        # ==========================================
        if not data or 'review' not in data:
            return jsonify({'error': 'Missing required parameter "review" in JSON body.'}), 400

        review_text = data['review']
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({'error': '"review" parameter must be a non-empty string.'}), 400

        # ==========================================
        # Step 10: Preprocess Raw Input Text in the API
        # ==========================================
        cleaned_review = preprocess_text(review_text)

        # ==========================================
        # Step 11: Vectorize Text using a TF-IDF Transformer
        # ==========================================
        vectorized_review = vectorizer.transform([cleaned_review])

        # ==========================================
        # Step 12: Make a Prediction with the Trained Model
        # ==========================================
        prediction = model.predict(vectorized_review)[0]  # Returns 'positive' or 'negative'
        probabilities = model.predict_proba(vectorized_review)[0]  # Probability of each class

        # Map predictions to classes
        classes = model.classes_
        class_probabilities = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        confidence = float(np.max(probabilities))

        # ==========================================
        # Step 13: Return a JSON Response from a Flask API
        # ==========================================
        return jsonify({
            'review': review_text,
            'cleaned_review': cleaned_review,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities
        })

    except Exception as e:
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

# ==========================================
# Step 14: Run the Flask Application Locally
# ==========================================
if __name__ == '__main__':
    # Start the local development server on host 0.0.0.0 and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
