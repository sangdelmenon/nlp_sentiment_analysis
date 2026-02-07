# Configuration for the NLP Sentiment Analysis Project

# Data parameters
NUM_REVIEWS = 2000
NUM_CATEGORIES = 4
NUM_TOPICS = 4

# Model paths
SENTIMENT_MODEL_PATH = "models/sentiment_model.joblib"
CATEGORY_MODEL_PATH = "models/category_model.joblib"
CATEGORY_ENCODER_PATH = "models/category_encoder.joblib"
LDA_MODEL_PATH = "models/lda_model.gensim"
LDA_DICTIONARY_PATH = "models/lda_dictionary.gensim"
TFIDF_VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

# TF-IDF parameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Word2Vec parameters
EMBEDDING_DIM = 300

# Logistic Regression parameters
MAX_ITER = 1000

# LDA parameters
LDA_PASSES = 10
NUM_WORDS = 10
