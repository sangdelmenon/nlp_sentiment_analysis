from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class TopicModeler:
    def __init__(self, num_topics=15):
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None

    def train_lda(self, tokenized_texts):
        """
        Trains an LDA model.
        """
        self.dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=10, random_state=42)
        
        print(f"LDA Model trained with {self.num_topics} topics.")
        return self.lda_model

    def get_topic_words(self, topic_id, num_words=10):
        """
        Returns the top words for a given topic.
        """
        if self.lda_model is None:
            raise Exception("LDA model not trained yet.")
        
        return self.lda_model.show_topic(topic_id, num_words)

    def visualize_topics(self, num_words=10):
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

if __name__ == '__main__':
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

