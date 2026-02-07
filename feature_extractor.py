from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import config

class FeatureExtractor:
    def __init__(self, max_features=config.MAX_FEATURES, embedding_dim=config.EMBEDDING_DIM):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=config.NGRAM_RANGE)
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

