# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def assign_labels(df):
    """
    Assign labels to tweets based on keywords (simplified for this example).

    Parameters:
    - df: DataFrame containing tweets.

    Returns:
    - DataFrame with an additional 'label' column.
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

def train_model(df):
    """
    Train the sentiment analysis model.

    Parameters:
    - df: DataFrame containing processed tweets and labels.

    Returns:
    - Trained model and vectorizer.
    """
    # Prepare data
    X = df['processed_text']
    y = df['label']
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Model training complete and saved.")
    return model, vectorizer

if __name__ == "__main__":
    df = pd.read_csv('processed_tweets.csv')
    df = assign_labels(df)
    model, vectorizer = train_model(df)
