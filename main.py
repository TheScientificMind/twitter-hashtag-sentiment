import os
import tweepy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import pickle
from dotenv import load_dotenv
from my_preprocess import preprocess_twt

# don"t run analysis before training the model (train.py)

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
    tweet_list = [preprocess_twt(twt) for twt in tweet_list]
    tweet_list = new_vectorizer(np.array(tweet_list))

    # makes sentiment prections
    predictions = model.predict(np.array(tweet_list))
    pdxn = np.mean(predictions, axis=0)
    indx = np.argmax(pdxn)
    sentiment_labels = ["negative", "neutral", "positive"]
    sentiment_pdxn = {sentiment_labels[i]: pdxn[i] for i in range(3)}

    print(f"Predicted sentiment: {sentiment_labels[indx]}")
    print(f"Sentiment chances: {sentiment_pdxn}")
except Exception as err:
    print("There was an error (likely a mistaken input):\n\n{err}\n\nPlease try again.")