# visualization.py

import pandas as pd
import matplotlib.pyplot as plt

def visualize_sentiment_distribution(df):
    """
    Visualize the sentiment distribution in the dataset.

    Parameters:
    - df: DataFrame containing the 'label' column.

    Displays:
    - Pie chart of sentiment distribution.
    """
    sentiment_counts = df['label'].value_counts()
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiment_counts.get(1, 0), sentiment_counts.get(0, 0), sentiment_counts.get(2, 0)]
    colors = ['green', 'red', 'grey']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('processed_tweets.csv')
    df = assign_labels(df)
    visualize_sentiment_distribution(df)
