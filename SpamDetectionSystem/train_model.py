import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK data (if not present)
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special chars and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (split by space)
    tokens = text.split()
    # Remove stopwords and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

def train():
    print("Loading data...")
    try:
        df = pd.read_csv("spam_data.csv")
    except FileNotFoundError:
        print("Error: spam_data.csv not found. Run data_generator.py first.")
        return

    print("Preprocessing data...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Split data
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF Vectorization
    print("Extracting features...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model and Vectorizer
    print("Saving model...")
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("Done! Model saved as 'spam_model.pkl' and vectorizer as 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    train()
