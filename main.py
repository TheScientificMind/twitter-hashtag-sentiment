import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt
import datetime

# import tweepy

# frequently used source: 
# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
# https://www.youtube.com/watch?v=hprBCp_UJN0&ab_channel=CodeHeroku

# hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

# # keeps asking until given a proper hashtag
# while not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
#     print("Your hashtag was invalid. Please input a new one.")
#     hashtag = input("What new hashtag would you like to analyze (e.g. #photography): ")

tweets = pd.read_csv("tweets_dataset.csv")

split_point = round(len(tweets) * 0.8) # the id at which the tweet dataset should be split

train = tweets[:split_point]
train_text, train_labels = train["text"], train["label"]

test = tweets[split_point:]
test_text, test_labels = test["text"], test["label"]

vectorizer = layers.TextVectorization(
    max_tokens = 2500,
    output_mode = "int", 
    output_sequence_length = 200,
    pad_to_max_tokens = True
)

print(train_text.dtypes)
exit()

vectorizer.adapt(train_text)
vectorizer.adapt(test_text)

model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length = 200),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16 ,activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()