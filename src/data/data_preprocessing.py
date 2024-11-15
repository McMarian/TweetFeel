# data_preprocessing.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import remove_emojis

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_text(text):
    """
    Clean the text by removing URLs, mentions, hashtags, emojis, and punctuation.

    Parameters:
    - text: The text to clean.

    Returns:
    - Cleaned text.
    """
    text = remove_emojis(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()                  # Convert to lowercase
    return text

def preprocess_text(text):
    """
    Tokenize, remove stopwords, and lemmatize the text.

    Parameters:
    - text: The text to preprocess.

    Returns:
    - Preprocessed text as a string.
    """
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_tweets(df):
    """
    Apply cleaning and preprocessing to a DataFrame of tweets.

    Parameters:
    - df: DataFrame containing tweets.

    Returns:
    - DataFrame with additional columns for cleaned and processed text.
    """
    df['clean_text'] = df['text'].apply(clean_text)
    df['processed_text'] = df['clean_text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('tweets.csv')
    df = preprocess_tweets(df)
    df.to_csv('processed_tweets.csv', index=False)
    print("Preprocessing complete. Saved to 'processed_tweets.csv'.")
