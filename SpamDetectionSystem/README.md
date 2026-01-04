# AI Twitter Spam Detection System 🐦

An advanced Machine Learning application that detects spam in **Raw Text** and **Twitter (X) Share Links**.
Built with Python, Scikit-Learn, and Streamlit.

## Features
- **📝 Text Analysis**: Type any text to classify it as Spam or Not Spam.
- **🔗 Link Analysis**: Paste a Twitter/X share link to automatically extract and analyze the tweet.
- **🧠 Explainability**: See which words triggered the spam detection.
- **🚀 Lightweight**: Runs locally without needing paid API keys (uses `ntscraper`).

## System Requirements
- Python 3.8 or higher
- Internet connection (for Link Analysis)

## How to Run (On any Laptop)

If you are moving this project to a new laptop, follow these steps:

### 1. Copy the Project
Copy the entire `SpamDetectionSystem` folder to the new computer.

### 2. Install Dependencies
Open a terminal (Command Prompt or PowerShell), navigate to the folder, and run:
```bash
pip install -r requirements.txt
```
*Note: This detects and installs all necessary libraries (streamlit, joblib, ntscraper, etc).*

### 3. Run the Application
In the same terminal, run:
```bash
streamlit run app.py
```
The app will open automatically in your browser (usually at `http://localhost:8501`).

## Troubleshooting

- **Link Analysis Failed?**: The system uses free Nitter instances to scrape tweets. If you see "Service Unavailable", just wait a few seconds and try again. This is normal for free scraping tools.
- **Model Not Found?**: Ensure you are running the command from inside sample folder or that `spam_model.pkl` is present next to `app.py`.

## Project Structure
- `app.py`: Main application code.
- `spam_model.pkl`: Pre-trained AI model.
- `tfidf_vectorizer.pkl`: Text processor.
- `spam_data.csv`: Dataset used for training.
- `train_model.py`: Script to retrain the model (optional).
