from gensim import corpora, models
import matplotlib.pyplot as plt
import os
import config

class TopicModeler:
    def __init__(self, num_topics=config.NUM_TOPICS):
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None

    def train_lda(self, tokenized_texts):
        """
        Trains an LDA model.
        """
        self.dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=config.LDA_PASSES, random_state=42)
        
        print(f"LDA Model trained with {self.num_topics} topics.")
        return self.lda_model

    def get_topic_words(self, topic_id, topn=config.NUM_WORDS):
        """
        Returns the top words for a given topic.
        """
        if self.lda_model is None:
            raise Exception("LDA model not trained yet.")
        
        return self.lda_model.show_topic(topic_id, topn=topn)

    def save_model(self, model_path=config.LDA_MODEL_PATH, dictionary_path=config.LDA_DICTIONARY_PATH):
        """Saves the LDA model and dictionary."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.lda_model.save(model_path)
        self.dictionary.save(dictionary_path)
        print(f"LDA model saved to {model_path} and {dictionary_path}")

    def load_model(self, model_path=config.LDA_MODEL_PATH, dictionary_path=config.LDA_DICTIONARY_PATH):
        """Loads the LDA model and dictionary."""
        self.lda_model = models.LdaModel.load(model_path)
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        print(f"LDA model loaded from {model_path} and {dictionary_path}")

    def visualize_topics(self, num_words=config.NUM_WORDS):
        """
        Generates bar charts for each topic's word distribution.
        """
        if self.lda_model is None:
            raise Exception("LDA model not trained yet.")
        
        cols = 2
        rows = (self.num_topics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True)
        axes = axes.flatten()

        for i in range(self.num_topics):
            topic_words = self.lda_model.show_topic(i, topn=num_words)
            words, probabilities = zip(*topic_words)
            
            ax = axes[i]
            ax.barh(words, probabilities)
            ax.set_title(f'Topic {i+1}')
            ax.invert_yaxis()

        # Hide any unused subplots
        for j in range(self.num_topics, len(axes)):
            axes[j].axis('off')

        fig.suptitle("Top Words per Topic", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

