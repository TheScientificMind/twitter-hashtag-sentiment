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

model = load_model("twitter_model")

# loads vectorizer, source: stackoverflow.com/questions/65103526
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_vectorizer = layers.TextVectorization(
    max_tokens=from_disk["config"]["max_tokens"],
    output_mode=from_disk["config"]["output_mode"],
    output_sequence_length=from_disk["config"]["output_sequence_length"]
)

new_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_vectorizer.set_weights(from_disk["weights"])

tweet = input("What tweet would you like to analyze: ")

# preprocesses tweets
tweet = [preprocess_twt(tweet)]
tweet = new_vectorizer(np.array(tweet))

# makes sentiment prections
predictions = model.predict(np.array(tweet))
pdxn = np.mean(predictions, axis=0)
indx = np.argmax(pdxn)
sentiment_labels = ["negative", "neutral", "positive"]
sentiment_pdxn = {sentiment_labels[i]: pdxn[i] for i in range(3)}

print(f"Predicted sentiment: {sentiment_labels[indx]}")
print(f"Sentiment chances: {sentiment_pdxn}")



