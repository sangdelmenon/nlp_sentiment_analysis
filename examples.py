"""
This file contains example usage for each of the modules in the project.
You can run this file to see the output of each module.
"""
from data_loader import ReviewDataLoader
from text_processor import TextPreprocessor
from feature_extractor import FeatureExtractor
from sentiment_analyzer import SentimentAnalyzer
from category_classifier import CategoryClassifier
from topic_modeler import TopicModeler
import pandas as pd

def run_data_loader_example():
    print("
" + "="*30)
    print("Data Loader Example")
    print("="*30)
    data_loader = ReviewDataLoader(num_reviews=1000)
    reviews_df = data_loader.generate_synthetic_reviews()
    data_loader.perform_eda(reviews_df)
    print("
Sample of generated data:")
    print(reviews_df.head())

def run_text_processor_example():
    print("\n" + "="*30)
    print("Text Processor Example")
    print("="*30)
    preprocessor = TextPreprocessor()
    sample_text = "This product is AMAZING!!! Works great :) I will NOT buy this again."
    processed_text = preprocessor.process_pipeline(sample_text)
    
    print(f"Original Text: '{sample_text}'")
    print(f"Processed Text: '{processed_text}'")

    # Example from instructions
    instruction_example = "This product is AMAZING!!! Works great :)"
    processed_instruction_example = preprocessor.process_pipeline(instruction_example)
    print(f"\nInstruction Example: '{instruction_example}'")
    print(f"Processed Instruction Example: '{processed_instruction_example}'")

def run_feature_extractor_example():
    print("\n" + "="*30)
    print("Feature Extractor Example")
    print("="*30)
    from data_loader import ReviewDataLoader
    from text_processor import TextPreprocessor

    # 1. Load and preprocess data
    data_loader = ReviewDataLoader(num_reviews=100)
    reviews_df = data_loader.generate_synthetic_reviews()
    
    preprocessor = TextPreprocessor()
    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocessor.process_pipeline)
    
    corpus = reviews_df['processed_text'].tolist()
    
    # 2. Initialize FeatureExtractor
    feature_extractor = FeatureExtractor()

    # 3. TF-IDF Example
    print("="*30)
    print("TF-IDF Example")
    print("="*30)
    tfidf_matrix = feature_extractor.extract_tfidf(corpus)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Sample feature names: {feature_extractor.tfidf_vectorizer.get_feature_names_out()[:10]}")
    
    # 4. Word2Vec Example
    print("\n" + "="*30)
    print("Word2Vec Example")
    print("="*30)
    sentences = [text.split() for text in corpus]
    w2v_model = feature_extractor.train_word2vec(sentences)
    print(f"Word2Vec model trained. Vocab size: {len(w2v_model.wv.index_to_key)}")
    sample_embedding = feature_extractor.get_word2vec_embeddings(['sample', 'review'])
    print(f"Sample Word2Vec embedding shape: {sample_embedding.shape}")

    # 5. BERT Example
    print("\n" + "="*30)
    print("BERT Example")
    print("="*30)
    sample_text_for_bert = "this is a sample review"
    bert_embedding = feature_extractor.extract_bert_embeddings(sample_text_for_bert)
    print(f"BERT embedding for '{sample_text_for_bert}':")
    print(f"Shape: {bert_embedding.shape}")

def run_sentiment_analyzer_example():
    print("\n" + "="*30)
    print("Sentiment Analyzer Example")
    print("="*30)
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

def run_category_classifier_example():
    print("\n" + "="*30)
    print("Category Classifier Example")
    print("="*30)
    from data_loader import ReviewDataLoader
    from text_processor import TextPreprocessor
    from feature_extractor import FeatureExtractor
    
    # 1. Load and preprocess data
    data_loader = ReviewDataLoader(num_reviews=1000, num_categories=5)
    reviews_df = data_loader.generate_synthetic_reviews()
    
    preprocessor = TextPreprocessor()
    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocessor.process_pipeline)
    
    # 2. Feature Extraction
    feature_extractor = FeatureExtractor()
    tfidf_matrix = feature_extractor.extract_tfidf(reviews_df['processed_text'])
    
    # 3. Category Classification
    category_classifier = CategoryClassifier()
    
    X = tfidf_matrix
    y = reviews_df['category']
    
    category_classifier.train_classifier(X, y)
    
    # Predict on a sample
    sample_text = "This is a review about a phone camera"
    processed_sample = preprocessor.process_pipeline(sample_text)
    sample_features = feature_extractor.tfidf_vectorizer.transform([processed_sample])
    
    prediction = category_classifier.predict_category(sample_features)
    print(f"\nPrediction for '{sample_text}': {prediction[0]}")

def run_topic_modeler_example():
    print("\n" + "="*30)
    print("Topic Modeler Example")
    print("="*30)
    from data_loader import ReviewDataLoader
    from text_processor import TextPreprocessor

    # 1. Load and preprocess data
    data_loader = ReviewDataLoader(num_reviews=500)
    reviews_df = data_loader.generate_synthetic_reviews()
    
    preprocessor = TextPreprocessor()
    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocessor.process_pipeline)
    tokenized_texts = [text.split() for text in reviews_df['processed_text']]

    # 2. Topic Modeling
    topic_modeler = TopicModeler(num_topics=5)
    topic_modeler.train_lda(tokenized_texts)

    # 3. Show top words for a topic
    print("\nTop words for Topic 1:")
    print(topic_modeler.get_topic_words(0))

    # 4. Visualize topics
    print("\nGenerating topic visualizations...")
    # This will open a plot window
    # topic_modeler.visualize_topics()
    print("[INFO] Visualization is commented out to prevent blocking script execution in a non-interactive environment.")
    print("[INFO] Uncomment the line `topic_modeler.visualize_topics()` to see the bar charts.")

if __name__ == '__main__':
    run_data_loader_example()
    run_text_processor_example()
    run_feature_extractor_example()
    run_sentiment_analyzer_example()
    run_category_classifier_example()
    run_topic_modeler_example()
