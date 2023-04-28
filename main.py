import os
import tweepy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import pickle
from dotenv import load_dotenv
from my_preprocess import Preprocessor

# don't run analysis before training the model (train.py)

# commonly used source: docs.tweepy.org

load_dotenv()

try:
    model = load_model("twitter_model")
    model.summary()

    # loads vectorizer, source: stackoverflow.com/questions/65103526
    from_disk = pickle.load(open("tv_layer.pkl", "rb"))
    new_vectorizer = layers.TextVectorization(
        max_tokens=from_disk["config"]["max_tokens"],
        output_mode=from_disk["config"]["output_mode"],
        output_sequence_length=from_disk["config"]["output_sequence_length"]
    )
    new_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_vectorizer.set_weights(from_disk["weights"])

    # gets twitter access credentials
    auth = tweepy.OAuth1UserHandler(
        os.getenv("api_key"), os.getenv("api_key_secret"),
        os.getenv("access_token"), os.getenv("access_token_secret")
    )

    # Create API object
    api = tweepy.API(auth)

    hashtag = input("What term would you like to analyze (e.g. #photography, @elonmusk, apple): ").lower().strip()

    # Collect tweets using the Cursor object
    tweets = tweepy.Cursor(api.search_tweets, hashtag, lang="en").items(100)

    # converts tweets to list
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.text)

    # preprocesses tweets
    tweet_list = [Preprocessor(twt=tweet).clean_twt() for tweet in tweet_list]
    tweet_list = new_vectorizer(np.array(tweet_list))

    # makes sentiment prections
    predictions = model.predict(np.array(tweet_list))
    predictions = [np.argmax(prediction) for prediction in predictions]
    unrounded_indx = np.mean(predictions)
    sentiment_labels = ["negative", "neutral", "positive"]

    print(f"Predicted sentiment: {sentiment_labels[round(unrounded_indx)]}")
    print(f"Sentiment value (0 = negative, 1 = neutral, 2 = positive): {unrounded_indx}")
except Exception as err:
    print(f"There was an error:\n\n{err}\n\nPlease try again.")