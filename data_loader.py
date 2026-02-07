import pandas as pd
import numpy as np

class ReviewDataLoader:
    def __init__(self, num_reviews=1000, num_categories=10):
        self.num_reviews = num_reviews
        self.num_categories = num_categories
        self.categories = [f'Category_{i+1}' for i in range(num_categories)]
        self.sentiments = ['Positive', 'Negative', 'Neutral']

    def generate_synthetic_reviews(self):
        """Generates a synthetic dataset of reviews."""
        print("Generating synthetic reviews...")
        data = {
            'review_id': range(self.num_reviews),
            'review_text': [f'This is a sample review text number {i+1}.' for i in range(self.num_reviews)],
            'category': np.random.choice(self.categories, self.num_reviews),
            'sentiment': np.random.choice(self.sentiments, self.num_reviews, p=[0.5, 0.3, 0.2]),
            'rating': np.random.randint(1, 6, self.num_reviews)
        }
        df = pd.DataFrame(data)
        return df

    def load_from_csv(self, file_path, text_column, sentiment_column, category_column=None):
        """
        Loads review data from a CSV file.
        
        :param file_path: Path to the CSV file.
        :param text_column: The name of the column containing the review text.
        :param sentiment_column: The name of the column containing the sentiment.
        :param category_column: The name of the column containing the category (optional).
        :return: A pandas DataFrame.
        """
        print(f"Loading reviews from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Rename columns for consistency
        df = df.rename(columns={
            text_column: 'review_text',
            sentiment_column: 'sentiment'
        })
        
        if category_column:
            df = df.rename(columns={category_column: 'category'})
        else:
            # If no category column, create a default one
            df['category'] = 'Default'
            
        return df[['review_text', 'sentiment', 'category']]

    def perform_eda(self, df):
        """Performs basic exploratory data analysis."""
        print("="*30)
        print("Exploratory Data Analysis")
        print("="*30)
        print("Dataset Info:")
        df.info()
        print("\nSentiment Distribution:")
        print(df['sentiment'].value_counts())
        print("\nCategory Distribution:")
        print(df['category'].value_counts())
        print("="*30)


