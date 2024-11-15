# main.py

from src.data.data_collection import collect_tweets
from src.data.data_preprocessing import preprocess_tweets, preprocess_text
from model_training import assign_labels, train_model
from model_evaluation import evaluate_model
from src.visualization.visualization import visualize_sentiment_distribution
from model.ensemble_classifier import SentimentEnsemble
from model.emotion_detector import EmotionDetector
from model.advanced_classifier import AdvancedSentimentClassifier
from model.sentiment_reasoner import SentimentReasoner
from model.feedback_learner import FeedbackLearner
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("\n🤖 Welcome to TweetFeel - Sentiment & Emotion Analysis! 🎭\n")
    
    # Initialize models
    sentiment_model = AdvancedSentimentClassifier()
    emotion_detector = EmotionDetector()
    sentiment_reasoner = SentimentReasoner()
    feedback_learner = FeedbackLearner()
    
    while True:
        try:
            choice = input("\n1. Analyze tweets by keyword\n2. Analyze single text\n3. Exit\nChoice: ")
            
            if choice == '3':
                print("\nThanks for using TweetFeel! Goodbye! 👋\n")
                break
                
            if choice == '1':
                analyze_tweets()
            elif choice == '2':
                analyze_single_text(sentiment_model, emotion_detector, sentiment_reasoner, feedback_learner)
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted. Goodbye! 👋\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Let's try again!")

def analyze_tweets():
    print("\n🤖 Welcome to Enhanced TweetFeel - Sentiment & Emotion Analysis! 🎭\n")
    
    while True:
        try:
            print("\nWhat would you like to analyze? 🤔")
            keyword = input("Enter keyword to search for tweets (or 'exit' to quit): ")
            
            if keyword.lower() == 'exit':
                print("\nThanks for using TweetFeel! Goodbye! 👋\n")
                sys.exit(0)
                
            count = input("How many tweets should I analyze? (default: 100): ")
            count = int(count) if count.strip() else 100
            
            print(f"\n🔍 Searching for {count} tweets about '{keyword}'...")
            tweets_df = collect_tweets(keyword, count)
            
            if tweets_df.empty:
                print("❌ No tweets found. Try a different keyword!")
                continue
                
            print(f"✅ Found {len(tweets_df)} tweets!")
            tweets_df.to_csv('output/tweets.csv', index=False)

            print("\n🧹 Cleaning up the tweets...")
            tweets_df = preprocess_tweets(tweets_df)
            tweets_df.to_csv('output/processed_tweets.csv', index=False)
            print("✅ Preprocessing complete!")

            print("\n🏷️ Analyzing sentiment...")
            tweets_df = assign_labels(tweets_df)
            tweets_df.to_csv('output/labeled_tweets.csv', index=False)
            print("✅ Analysis complete!")

            print("\n📊 Generating visualization...")
            visualize_sentiment_distribution(tweets_df)

            print("\n🤖 Training model...")
            model, vectorizer = train_model(tweets_df)

            print("\n📋 Evaluating results...")
            evaluate_model()
            
            print("\n✨ Analysis complete! Check the 'output' folder for detailed results!")
            
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted. Goodbye! 👋\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Let's try again!")

def analyze_single_text(sentiment_model, emotion_detector, sentiment_reasoner, feedback_learner):
    print("\n🤖 Welcome to Enhanced TweetFeel - Sentiment & Emotion Analysis! 🎭\n")
    
    while True:
        try:
            text = input("\nEnter text to analyze (or 'exit' to quit): ")
            
            if text.lower() == 'exit':
                break
            
            # Process text
            processed_text = preprocess_text(text)
            
            # Get sentiment and emotion
            sentiment = sentiment_model.predict([processed_text])[0]
            emotion = emotion_detector.detect_emotion(processed_text)
            
            # Display results
            print("\n📊 Analysis Results:")
            print(f"Sentiment: {['Negative', 'Positive', 'Neutral'][sentiment]}")
            print(f"Emotion: {emotion.capitalize()}")
            
            # Visualize results
            plot_results(sentiment, emotion)

            # Analyze sentiment
            analysis = analyze_sentiment(text, sentiment_model, sentiment_reasoner)
            print("\n📊 Analysis Results:")
            print(f"Sentiment: {analysis['final_sentiment']}")
            print(f"Confidence: {analysis['confidence']:.2f}")
            print(f"Reasoning: {analysis['reason']}")

            # Feedback
            feedback = input("Was this analysis correct? (y/n): ")
            if feedback.lower() == 'n':
                actual = input("What was the correct sentiment? (0:negative, 1:positive, 2:neutral): ")
                feedback_learner.store_feedback(text, sentiment, int(actual))
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted. Goodbye! 👋\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Let's try again!")

def plot_results(sentiment, emotion):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Sentiment plot
    sns.barplot(x=['Negative', 'Positive', 'Neutral'], 
                y=[1 if sentiment == i else 0 for i in range(3)],
                ax=ax1)
    ax1.set_title('Sentiment Analysis')
    
    # Emotion plot
    emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'neutral']
    sns.barplot(x=emotions,
                y=[1 if emotion == e else 0 for e in emotions],
                ax=ax2)
    ax2.set_title('Emotion Detection')
    plt.tight_layout()
    plt.show()

def analyze_sentiment(text, model, reasoner):
    # Get model prediction
    prediction = model.predict([text])[0]
    
    # Get reasoning
    analysis = reasoner.analyze(text, prediction)
    
    print("\n📊 Analysis Results:")
    print(f"Sentiment: {analysis['final_sentiment']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Reasoning: {analysis['reason']}")
    
    return analysis

if __name__ == "__main__":
    main()

