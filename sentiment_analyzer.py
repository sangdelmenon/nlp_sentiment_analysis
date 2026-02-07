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


if __name__ == '__main__':
    from data_loader import ReviewDataLoader
    from text_processor import TextPreprocessor
    from feature_extractor import FeatureExtractor
    import pandas as pd

    # 1. Load and preprocess data
    data_loader = ReviewDataLoader(num_reviews=1000)
    reviews_df = data_loader.generate_synthetic_reviews()
    
    preprocessor = TextPreprocessor()
    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocessor.process_pipeline)
    
    # 2. Feature Extraction
    feature_extractor = FeatureExtractor()
    tfidf_matrix = feature_extractor.extract_tfidf(reviews_df['processed_text'])

    # 3. Sentiment Analysis
    sentiment_analyzer = SentimentAnalyzer()

    # VADER Example
    print("="*30)
    print("VADER Sentiment Analysis")
    print("="*30)
    sample_review = "This is a great product, I love it!"
    vader_scores = sentiment_analyzer.vader_sentiment(sample_review)
    print(f"VADER scores for '{sample_review}': {vader_scores}")
    
    # ML Classifier Example
    print("\n" + "="*30)
    print("Machine Learning Classifier (Logistic Regression)")
    print("="*30)
    # Map sentiments to numerical labels for training
    sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)
    
    X = tfidf_matrix
    y = reviews_df['sentiment_label']
    
    sentiment_analyzer.train_ml_classifier(X, y)
    
    # Predict on a sample
    sample_features = feature_extractor.tfidf_vectorizer.transform([preprocessor.process_pipeline("This is terrible")])
    prediction = sentiment_analyzer.predict_sentiment_ml(sample_features)
    reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
    print(f"Prediction for 'This is terrible': {reverse_mapping[prediction[0]]}")
    
    # BERT Placeholder
    sentiment_analyzer.fine_tune_bert_placeholder()
