import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ntscraper import Nitter
import random
import time

# Same preprocessing function to ensure consistency
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

import os

# Load resources
@st.cache_resource
def load_model():
    # Use absolute path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "spam_model.pkl")
    vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def get_tweet_text(url):
    """Extracts tweet text from a Twitter/X URL using ntscraper."""
    try:
        # Extract ID
        match = re.search(r'(twitter|x)\.com\/+\w+\/status\/(\d+)', url)
        if not match:
            return None, "Invalid Twitter/X URL format."
        
        tweet_id = match.group(2)
        
        # Scrape
        # We try to initialize Nitter. If it fails to find instances, it might raise "Cannot choose from empty sequence"
        try:
            scraper = Nitter(log_level=1)
            # Fetch tweet using 'terms' search for the ID
            results = scraper.get_tweets(tweet_id, mode='term', number=1)
        except Exception as e:
            return None, f"Service Unavailable: Could not connect to any Nitter instance. ({str(e)})"
            
        if results and 'tweets' in results and len(results['tweets']) > 0:
             return results['tweets'][0]['text'], None
        else:
             return None, "Tweet found but could not extract text (or tweet is media-only/deleted)."

    except Exception as e:
        return None, f"Scraping error: {str(e)}"

# Logic for prediction to be reused
def analyze_spam(text_content):
    try:
        model, vectorizer = load_model()
        
        # Process & Predict
        clean_text = preprocess_text(text_content)
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # Display Results
        if prediction == "spam":
            st.error(f"🚨 This is **SPAM** (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ This is **NOT SPAM** (Confidence: {proba[0]*100:.1f}%)")
            
        # Explainability
        st.markdown("---")
        st.subheader("Why?")
        feature_names = vectorizer.get_feature_names_out()
        dense_features = features.todense().tolist()[0]
        relevant_words = [(feature_names[i], score) for i, score in enumerate(dense_features) if score > 0]
        relevant_words.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_words:
            st.write("Top influential words in this tweet:")
            st.write(", ".join([f"**{word}**" for word, score in relevant_words[:5]]))
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you have trained the model first!")

st.set_page_config(page_title="Spam Detector", page_icon="🐦")

st.title("🐦 AI Twitter Spam Detector")
st.markdown("Enter a tweet below to check if it's **Spam** or **Not Spam**.")

tab1, tab2 = st.tabs(["📝 Analyze Text", "🔗 Analyze Link"])

with tab1:
    user_input = st.text_area("Tweet Text", placeholder="Ex: Win free iPhone now! Click here...")
    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            analyze_spam(user_input)

with tab2:
    st.info("⚠️ Note: This uses a free scraping method. If it fails, please wait a moment and try again.")
    url_input = st.text_input("Tweet Link", placeholder="https://x.com/username/status/123456789")
    if st.button("Analyze Link"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching tweet from Nitter (this may take a moment)..."):
                text, error = get_tweet_text(url_input)
            
            if error:
                st.error(error)
            elif text:
                st.info(f"**Extracted Text:**\n> {text}")
                analyze_spam(text)
            else:
                st.error("Could not extract text.")

st.markdown("---")
st.caption("Powered by Scikit-Learn & Nitter Scraper")
