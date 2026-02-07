from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import joblib
import os

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.ml_model = LogisticRegression(max_iter=1000)

    def vader_sentiment(self, text):
        """
        Analyzes sentiment using VADER.
        Returns a dictionary with 'pos', 'neu', 'neg', 'compound' scores.
        """
        return self.vader_analyzer.polarity_scores(text)

    def train_ml_classifier(self, X, y):
        """
        Trains a machine learning classifier for sentiment analysis.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ML Classifier trained. Test accuracy: {accuracy:.4f}")

    def predict_sentiment_ml(self, features):
        """
        Predicts sentiment using the trained ML model.
        """
        return self.ml_model.predict(features)

    def save_model(self, path="models/sentiment_model.joblib"):
        """Saves the trained ML model to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.ml_model, path)
        print(f"Sentiment model saved to {path}")

    def load_model(self, path="models/sentiment_model.joblib"):
        """Loads a trained ML model from a file."""
        self.ml_model = joblib.load(path)
        print(f"Sentiment model loaded from {path}")

    def fine_tune_bert_placeholder(self):
        """
        Placeholder for the BERT fine-tuning process.
        This is a complex task that requires significant setup.
        """
        print("\n[INFO] BERT fine-tuning is a complex process requiring a separate, detailed script.")
        print("This function is a placeholder. See `transformers` library documentation for examples.")


