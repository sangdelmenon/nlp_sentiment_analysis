import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Customizing stop words based on the instructions
        self.stop_words.discard('not')
        self.stop_words.update(['product', 'review', 'item'])


    def clean_text(self, text):
        """Removes special characters and lowercases text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def tokenize(self, text):
        """Tokenizes the text."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Removes stopwords from a list of tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens):
        """Lemmatizes a list of tokens."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def process_pipeline(self, text):
        """Runs the complete text preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

if __name__ == '__main__':
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

