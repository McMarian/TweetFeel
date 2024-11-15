# model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate_model():
    """
    Evaluate the trained model using the test data.
    """
    # Load data
    df = pd.read_csv('processed_tweets.csv')
    df = assign_labels(df)

    # Prepare data
    X = df['processed_text']
    y = df['label']
    vectorizer = joblib.load('vectorizer.pkl')
    X_vectorized = vectorizer.transform(X)

    # Load model
    model = joblib.load('sentiment_model.pkl')

    # Predict
    y_pred = model.predict(X_vectorized)

    # Evaluation report
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive', 'Neutral']))

def assign_labels(df):
    """
    Same as in model_training.py (imported here for completeness).
    """
    positive_keywords = ['good', 'great', 'happy', 'love', 'excellent', 'awesome', 'fantastic', 'best']
    negative_keywords = ['bad', 'terrible', 'sad', 'hate', 'worst', 'awful', 'horrible', 'poor']
    labels = []
    for text in df['processed_text']:
        if any(word in text for word in positive_keywords):
            labels.append(1)  # Positive
        elif any(word in text for word in negative_keywords):
            labels.append(0)  # Negative
        else:
            labels.append(2)  # Neutral
    df['label'] = labels
    return df

if __name__ == "__main__":
    evaluate_model()
