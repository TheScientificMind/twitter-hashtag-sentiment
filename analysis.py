import tweepy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import pickle
from my_preprocess import preprocess_twt, vectorize_twts

model = load_model('twitter_model')

# Next 9 lines source: stackoverflow.com/questions/65103526
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_vectorizer = layers.TextVectorization(
    max_tokens=from_disk['config']['max_tokens'],
    output_mode=from_disk['config']['output_mode'],
    output_sequence_length=from_disk['config']['output_sequence_length']
)

new_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_vectorizer.set_weights(from_disk['weights'])

hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

# keeps asking until given a proper hashtag
while not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
    print("Your hashtag was invalid. Please input a new one.")
    hashtag = input("What new hashtag would you like to analyze (e.g. #photography): ")

