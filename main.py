from data_loader import ReviewDataLoader
from text_processor import TextPreprocessor
from feature_extractor import FeatureExtractor
from sentiment_analyzer import SentimentAnalyzer
from category_classifier import CategoryClassifier
from topic_modeler import TopicModeler
import pandas as pd
import joblib

class SentimentPipeline:
    def __init__(self, num_reviews=1000, num_categories=5, num_topics=5):
        self.num_reviews = num_reviews
        self.num_categories = num_categories
        self.num_topics = num_topics
        
        # Initialize components
        self.data_loader = ReviewDataLoader(num_reviews=self.num_reviews, num_categories=self.num_categories)
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.category_classifier = CategoryClassifier()
        self.topic_modeler = TopicModeler(num_topics=self.num_topics)

    def run_complete_pipeline(self):
        """
        Runs the end-to-end NLP pipeline.
        """
        print("Starting NLP Pipeline...")

        # 1. Load Data
        print("\n[STEP 1] Loading and generating synthetic data...")
        reviews_df = self.data_loader.generate_synthetic_reviews()
        self.data_loader.perform_eda(reviews_df)

        # 2. Preprocess Text
        print("\n[STEP 2] Preprocessing text data...")
        reviews_df['processed_text'] = reviews_df['review_text'].apply(self.text_preprocessor.process_pipeline)
        print("Text preprocessing complete.")
        print(reviews_df[['review_text', 'processed_text']].head())

        # 3. Feature Extraction
        print("\n[STEP 3] Extracting features...")
        corpus = reviews_df['processed_text'].tolist()
        
        # TF-IDF
        print("Extracting TF-IDF features...")
        tfidf_matrix = self.feature_extractor.extract_tfidf(corpus)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Word2Vec
        print("Training Word2Vec model...")
        sentences = [text.split() for text in corpus]
        self.feature_extractor.train_word2vec(sentences)
        print("Word2Vec model trained.")

        # 4. Train Models
        print("\n[STEP 4] Training models...")
        
        # Sentiment Analysis ML Model
        print("\n--- Training Sentiment Analysis Model ---")
        sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)
        self.sentiment_analyzer.train_ml_classifier(tfidf_matrix, reviews_df['sentiment_label'])

        # Category Classification Model
        print("\n--- Training Category Classification Model ---")
        self.category_classifier.train_classifier(tfidf_matrix, reviews_df['category'])

        # Topic Modeling
        print("\n--- Training Topic Model (LDA) ---")
        self.topic_modeler.train_lda(sentences)
        for i in range(self.num_topics):
            print(f"Topic {i+1}: {self.topic_modeler.get_topic_words(i, topn=5)}")

        # 5. Analyze a sample review
        print("\n[STEP 5] Analyzing a new sample review...")
        sample_review_text = "The camera on this phone is fantastic, but the battery life is a disappointment."
        self.analyze_review(sample_review_text)

        print("\nNLP Pipeline finished.")

    def analyze_review(self, review_text):
        """
        Analyzes a single review using the trained models.
        """
        print(f"\nAnalyzing review: '{review_text}'")

        # Preprocess
        processed_text = self.text_preprocessor.process_pipeline(review_text)
        
        # VADER Sentiment
        vader_result = self.sentiment_analyzer.vader_sentiment(review_text)
        print(f"VADER Sentiment: {vader_result}")

        # TF-IDF Features for ML models
        features_tfidf = self.feature_extractor.tfidf_vectorizer.transform([processed_text])

        # ML Sentiment Prediction
        ml_sentiment_pred = self.sentiment_analyzer.predict_sentiment_ml(features_tfidf)
        sentiment_mapping_rev = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
        print(f"ML Sentiment Prediction: {sentiment_mapping_rev[ml_sentiment_pred[0]]}")

        # Category Prediction
        category_pred = self.category_classifier.predict_category(features_tfidf)
        print(f"Predicted Category: {category_pred[0]}")

        # Topic Distribution
        bow = self.topic_modeler.dictionary.doc2bow(processed_text.split())
        topics = self.topic_modeler.lda_model.get_document_topics(bow)
        print("Predicted Topics:")
        for topic_num, prop in topics:
            print(f"  - Topic {topic_num+1}: {prop*100:.2f}%")

    def save_models(self):
        """Saves all trained models."""
        print("\n[STEP 6] Saving all models...")
        self.sentiment_analyzer.save_model()
        self.category_classifier.save_model()
        self.topic_modeler.save_model()
        # Also save the tfidf vectorizer
        joblib.dump(self.feature_extractor.tfidf_vectorizer, "models/tfidf_vectorizer.joblib")
        print("All models saved in the 'models' directory.")

    def load_models(self):
        """Loads all trained models."""
        print("\nLoading all models...")
        self.sentiment_analyzer.load_model()
        self.category_classifier.load_model()
        self.topic_modeler.load_model()
        # Also load the tfidf vectorizer
        self.feature_extractor.tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        print("All models loaded.")


if __name__ == '__main__':
    pipeline = SentimentPipeline(num_reviews=2000, num_categories=4, num_topics=4)
    pipeline.run_complete_pipeline()
    pipeline.save_models()

    # Example of loading models and predicting
    print("\n" + "="*30)
    print("Example of loading models and predicting on a new review")
    print("="*30)
    new_pipeline = SentimentPipeline()
    new_pipeline.load_models()
    new_pipeline.analyze_review("This is a fantastic product, I'm so happy with it!")
