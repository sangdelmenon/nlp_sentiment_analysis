from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self, max_features=5000, embedding_dim=300):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        self.word2vec_model = None
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def extract_tfidf(self, corpus):
        """Extracts TF-IDF features from a corpus of text."""
        return self.tfidf_vectorizer.fit_transform(corpus)

    def train_word2vec(self, sentences, sg=1, window=5, min_count=2, workers=4):
        """Trains a Word2Vec model."""
        self.word2vec_model = Word2Vec(sentences, vector_size=self.embedding_dim, sg=sg, 
                                       window=window, min_count=min_count, workers=workers)
        return self.word2vec_model

    def get_word2vec_embeddings(self, tokens):
        """Gets the average Word2Vec embedding for a list of tokens."""
        if self.word2vec_model is None:
            raise Exception("Word2Vec model not trained yet.")
        
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.embedding_dim)
        return np.mean(vectors, axis=0)

    def extract_bert_embeddings(self, text):
        """Extracts BERT embeddings for a given text."""
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        # Use the embedding of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy()

if __name__ == '__main__':
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
