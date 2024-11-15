# data_collection.py

import tweepy
import pandas as pd
import config  # A separate file where API keys are stored securely

def collect_tweets(keyword, count=100):
    """
    Collect tweets containing the specified keyword.

    Parameters:
    - keyword: The keyword to search for.
    - count: Number of tweets to collect.

    Returns:
    - A Pandas DataFrame containing the tweets.
    """
    try:
        # Authenticate with the Twitter API
        auth = tweepy.OAuthHandler(config.API_KEY, config.API_SECRET_KEY)
        auth.set_access_token(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # Collect tweets
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(count)

        # Extract tweet information
        tweet_list = []
        for tweet in tweets:
            if 'retweeted_status' in dir(tweet):
                content = tweet.retweeted_status.full_text
            else:
                content = tweet.full_text
            tweet_list.append({'text': content, 'created_at': tweet.created_at, 'user': tweet.user.screen_name})

        # Create DataFrame
        df = pd.DataFrame(tweet_list)
        return df

    except Exception as e:
        print("An error occurred while collecting tweets:", e)
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    keyword = input("Enter keyword to search for tweets: ")
    count = int(input("Enter number of tweets to collect: "))
    tweets_df = collect_tweets(keyword, count)
    if not tweets_df.empty:
        tweets_df.to_csv('tweets.csv', index=False)
        print(f"Collected {len(tweets_df)} tweets containing the keyword '{keyword}'.")
    else:
        print("No tweets collected.")
