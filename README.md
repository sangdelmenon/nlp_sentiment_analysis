# NLP Sentiment Analysis Project 🚀

This project is a comprehensive NLP system for analyzing customer reviews. It performs sentiment analysis, category classification, and topic modeling on a dataset of reviews. This is a great project to showcase your NLP skills!

## Features ✨

- **Sentiment Analysis:** Utilizes VADER for rule-based sentiment scoring and a machine learning model (Logistic Regression) for trained sentiment prediction.
- **Category Classification:** Classifies reviews into different product categories using a Naive Bayes classifier.
- **Topic Modeling:** Discovers hidden topics in the reviews using Latent Dirichlet Allocation (LDA).
- **Model Persistence:** Trained models can be saved and loaded for later use, so you don't have to retrain them every time.

## Project Structure 📂

```
├── main.py                 # Main pipeline orchestration
├── data_loader.py          # Loads and generates data
├── text_processor.py       # Text preprocessing
├── feature_extractor.py    # Feature extraction (TF-IDF, Word2Vec, BERT)
├── sentiment_analyzer.py   # Sentiment analysis models
├── category_classifier.py  # Category classification models
├── topic_modeler.py        # Topic modeling
├── examples.py             # Example usage of each module
├── requirements.txt        # Project dependencies
├── .gitignore              # Files to ignore in git
└── README.md               # This file
```

## Setup ⚙️

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sangdelmenon/nlp_sentiment_analysis.git
    cd nlp_sentiment_analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Download NLTK data:** The first time you run the project, it will download the necessary NLTK data.

## Usage 🚀

To run the complete pipeline, including training and saving the models, run the `main.py` script:

```bash
python3 main.py
```

This will:
1.  Generate a synthetic dataset of reviews.
2.  Preprocess the text data.
3.  Extract features.
4.  Train the sentiment analysis, category classification, and topic models.
5.  Save the trained models to the `models/` directory.
6.  Run an example of loading the models and predicting on a new review.

To see examples of each module, you can run the `examples.py` script:
```bash
python3 examples.py
```

## Model Performance 📊

The models are trained on a small, synthetic dataset, so the performance is not expected to be high. Here are the results from a sample run:

-   **Sentiment Analysis (Logistic Regression):** Test accuracy: ~48%
-   **Category Classification (Naive Bayes):** Weighted F1-score: ~10%

To get better results, you can replace the synthetic data in `data_loader.py` with a real-world dataset.
