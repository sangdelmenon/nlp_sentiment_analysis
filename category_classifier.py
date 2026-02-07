from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import config

class CategoryClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        self.label_encoder = LabelEncoder()

    def train_classifier(self, X, y):
        """
        Trains a classifier to predict product categories.
        """
        # Encode string labels to numbers
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Category Classifier (Naive Bayes) trained. Weighted F1-score: {f1:.4f}")
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    def predict_category(self, features):
        """
        Predicts the category for a given set of features.
        Returns the category name.
        """
        prediction_encoded = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction_encoded)

    def save_model(self, model_path="models/category_model.joblib", encoder_path="models/category_encoder.joblib"):
        """Saves the trained model and label encoder."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Category model saved to {model_path} and {encoder_path}")

    def load_model(self, model_path="models/category_model.joblib", encoder_path="models/category_encoder.joblib"):
        """Loads a trained model and label encoder."""
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"Category model loaded from {model_path} and {encoder_path}")

