import config
from data_loader import ReviewDataLoader
from text_processor import TextPreprocessor
from feature_extractor import FeatureExtractor
from sentiment_analyzer import SentimentAnalyzer
from category_classifier import CategoryClassifier
from topic_modeler import TopicModeler
import pandas as pd
import joblib

class SentimentPipeline:
    """
    Orchestrates the entire NLP pipeline, from data loading to model training and analysis.
    """
    def __init__(self, num_reviews=config.NUM_REVIEWS, num_categories=config.NUM_CATEGORIES, num_topics=config.NUM_TOPICS):
        """
        Initializes the pipeline with the specified parameters.
        """
        self.num_reviews = num_reviews
        self.num_categories = num_categories
        self.num_topics = num_topics
        
        # Initialize each component of the NLP pipeline
        self.data_loader = ReviewDataLoader(num_reviews=self.num_reviews, num_categories=self.num_categories)
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(max_features=config.MAX_FEATURES, embedding_dim=config.EMBEDDING_DIM)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.category_classifier = CategoryClassifier()
        self.topic_modeler = TopicModeler(num_topics=self.num_topics)

    def run_complete_pipeline(self, use_csv=False, csv_path=None, text_col=None, sentiment_col=None, cat_col=None):
        """
        Runs the end-to-end NLP pipeline.
        This includes loading data, preprocessing, feature extraction, training, and analysis.
        
        :param use_csv: If True, loads data from a CSV file. Otherwise, generates synthetic data.
        :param csv_path: Path to the CSV file.
        :param text_col: The name of the column containing the review text.
        :param sentiment_col: The name of the column containing the sentiment.
        :param cat_col: The name of the column containing the category (optional).
        """
        print("Starting NLP Pipeline...")

        # Step 1: Load and explore the data
        if use_csv:
            print(f"\n[STEP 1] Loading data from {csv_path}...")
            reviews_df = self.data_loader.load_from_csv(csv_path, text_col, sentiment_col, cat_col)
        else:
            print("\n[STEP 1] Loading and generating synthetic data...")
            reviews_df = self.data_loader.generate_synthetic_reviews()
        
        self.data_loader.perform_eda(reviews_df)

        # Step 2: Preprocess the review text
        print("\n[STEP 2] Preprocessing text data...")
        reviews_df['processed_text'] = reviews_df['review_text'].apply(self.text_preprocessor.process_pipeline)
        print("Text preprocessing complete.")
        print(reviews_df[['review_text', 'processed_text']].head())

        # Step 3: Extract numerical features from the text
        print("\n[STEP 3] Extracting features...")
        corpus = reviews_df['processed_text'].tolist()
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        tfidf_matrix = self.feature_extractor.extract_tfidf(corpus)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Train Word2Vec model for word embeddings
        print("Training Word2Vec model...")
        sentences = [text.split() for text in corpus]
        self.feature_extractor.train_word2vec(sentences)
        print("Word2Vec model trained.")

        # Step 4: Train all the different models
        print("\n[STEP 4] Training models...")
        
        # Train the sentiment analysis model
        print("\n--- Training Sentiment Analysis Model ---")
        sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)
        self.sentiment_analyzer.train_ml_classifier(tfidf_matrix, reviews_df['sentiment_label'])

        # Train the category classification model
        print("\n--- Training Category Classification Model ---")
        self.category_classifier.train_classifier(tfidf_matrix, reviews_df['category'])

        # Train the topic model
        print("\n--- Training Topic Model (LDA) ---")
        self.topic_modeler.train_lda(sentences)
        for i in range(self.num_topics):
            print(f"Topic {i+1}: {self.topic_modeler.get_topic_words(i, topn=config.NUM_WORDS)}")

        # Step 5: Analyze a sample review with the trained models
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
        self.sentiment_analyzer.save_model(config.SENTIMENT_MODEL_PATH)
        self.category_classifier.save_model(config.CATEGORY_MODEL_PATH, config.CATEGORY_ENCODER_PATH)
        self.topic_modeler.save_model(config.LDA_MODEL_PATH, config.LDA_DICTIONARY_PATH)
        # Also save the tfidf vectorizer
        joblib.dump(self.feature_extractor.tfidf_vectorizer, config.TFIDF_VECTORIZER_PATH)
        print("All models saved in the 'models' directory.")

    def load_models(self):
        """Loads all trained models."""
        print("\nLoading all models...")
        self.sentiment_analyzer.load_model(config.SENTIMENT_MODEL_PATH)
        self.category_classifier.load_model(config.CATEGORY_MODEL_PATH, config.CATEGORY_ENCODER_PATH)
        self.topic_modeler.load_model(config.LDA_MODEL_PATH, config.LDA_DICTIONARY_PATH)
        # Also load the tfidf vectorizer
        self.feature_extractor.tfidf_vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
        print("All models loaded.")


if __name__ == '__main__':
    # To run with synthetic data:
    pipeline = SentimentPipeline()
    pipeline.run_complete_pipeline()
    pipeline.save_models()

    # # To run with data from a CSV file, uncomment the following lines:
    # # Make sure to replace 'your_dataset.csv' and the column names with your actual data.
    # pipeline_csv = SentimentPipeline()
    # pipeline_csv.run_complete_pipeline(
    #     use_csv=True,
    #     csv_path='your_dataset.csv',
    #     text_col='review_text_column_name',
    #     sentiment_col='sentiment_column_name',
    #     cat_col='category_column_name' # Optional
    # )
    # pipeline_csv.save_models()


    # Example of loading models and predicting
    print("\n" + "="*30)
    print("Example of loading models and predicting on a new review")
    print("="*30)
    new_pipeline = SentimentPipeline()
    new_pipeline.load_models()
    new_pipeline.analyze_review("This is a fantastic product, I'm so happy with it!")
