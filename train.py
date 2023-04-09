import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime
from datasets import load_dataset
from my_preprocess import preprocess_tweet, vectorize_tweet

# frequently used source: 
# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
# https://www.youtube.com/watch?v=hprBCp_UJN0&ab_channel=CodeHeroku

tweets = load_dataset("carblacac/twitter-sentiment-analysis", "None")

split_point = round(len(tweets) * 0.95) # the id at which the tweet dataset should be split

train = tweets[:split_point]
train_text, train_labels = train["text"].astype(str).adapt(preprocess_tweet()), train["label"].astype(int)

test = tweets[split_point:]
test_text, test_labels = test["text"].astype(str).adapt(preprocess_tweet()), test["label"].astype(int)

vectorize_tweet(train_text)
vectorize_tweet(test_text)

# building the model
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=200),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16 ,activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# compiling the model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
) 

# training the model
history = model.fit(
    train_text, 
    train_labels, 
    epochs=25, 
    batch_size=512, 
    validation_data=(test_text, test_labels)
)

loss, accuracy = model.evaluate(test_text, test_labels) # evaluating model loss/accuracy

model.save('twitter_model') # saves the model