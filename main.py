import os
import tweepy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import pickle
import time
from dotenv import load_dotenv
from my_preprocess import Preprocessor # importing my own preprocessing class

# don't run analysis before training the model (train.py)

# commonly used source: docs.tweepy.org

load_dotenv()

try:
    model = load_model("twitter_model")
    model.summary()

    # loads vectorizer
    # next 8 lines source: stackoverflow.com/questions/65103526
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

    tweet_num = 250
    run_loop = True
    while run_loop:
        time.sleep(.5)

        prompt = "What term would you like to analyze (e.g. #cutedogs, @elonmusk, terrible): "
        hashtag = input(prompt)
        hashtag = hashtag.lower().strip()

        # Collect tweets using the Cursor object
        tweets = tweepy.Cursor(api.search_tweets, hashtag, lang="en").items(tweet_num)

        # converts tweets to list
        tweet_list = []
        for tweet in tweets:
            tweet_list.append(tweet.text)

        # ensures that the API successfully  got enough tweets
        if len(tweet_list) < tweet_num:
            print("There was an issue getting those tweets.")
            restart_loop = input("Would you like to analyze a new term (y/n): ")
            restart_loop = restart_loop.lower().strip()
            if restart_loop == "y":
                continue
            else:
                break

        # preprocesses tweets
        tweet_list = [Preprocessor(twt=tweet).clean_twt() for tweet in tweet_list]
        tweet_list = new_vectorizer(np.array(tweet_list)) # vectorizes tweets

        # makes sentiment prections
        predictions = model.predict(np.array(tweet_list))
        predictions = [np.argmax(prediction) for prediction in predictions]
        indx = round(np.mean(predictions), 2) # gets the average of the predictions
        sentiment_labels = ["negative", "neutral", "positive"]

        print(f"Predicted sentiment: {sentiment_labels[round(indx)]}")
        print(f"Sentiment value (0 = negative, 1 = neutral, 2 = positive): {indx}")

        # stop the loop if the user doesn't want to analyze another term
        if input("Would you like to analyze another term (y/n): ").lower().strip() == "n":
            run_loop = False
except Exception as err:
    print(f"There was an error:\n\n{err}\n\nPlease try again.")