import sys
import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses
from keras.models import load_model

from tensorboard import notebook

import matplotlib.pyplot as plt
import datetime

# import seaborn as sns
# import spacy
# import re
# import tweepy
# import pydot
# import graphviz

# frequently used source: https://www.tensorflow.org/text/tutorials/text_classification_rnn
# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
# https://www.youtube.com/watch?v=hprBCp_UJN0&ab_channel=CodeHeroku

def preprocess(x):
    return text_vectorizer(x)

# ensures the file isn't being imported as a module
if __name__ == "__main__":
    print("Version: ", tf.__version__)
    df = pd.read_csv("tweetsdataset.csv")
    df.sample(frac = 1)
    df.reset_index(drop = True)
    
    train = df[:round(len(df) * 0.8)]
    val = df[round(len(df) * 0.8):round(len(df) * 0.9)]
    test = df[round(len(df) * 0.9):]

    hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

    # invalid hashtag causes program to stop
    if not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
        print("Your hashtag was invalid. Please run the program and try again.")
        sys.exit()

    text_vectorizer = layers.TextVectorization(
        output_mode='multi_hot', 
        max_tokens=2500
        )

    # features = train.map(lambda x, y: x)
    features = df["text"].values.tolist()

    text_vectorizer.adapt(features)

    inputs = keras.Input(shape=(1,), dtype='string')

    outputs = layers.Dense(1)(preprocess(inputs))

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    
    model.summary()
    # keras.utils.plot_model(model, "sentiment_classifier.png")
    # keras.utils.plot_model(model, "sentiment_classifier_with_shape_info.png", show_shapes=True)

    # Create TensorBoard folders
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create callbacks
    my_callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    ]

    epochs = 10

    history = model.fit(
        tf.data.Dataset.from_tensor_slices((train['text'].values, train['label'].values)).shuffle(buffer_size=10000).batch(512),
        epochs=epochs,
        validation_data=val.batch(512),
        callbacks=my_callbacks,
        verbose=1)
    
    len(history.history['loss']) 

    results = model.evaluate(test.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    history_dict = history.history
    history_dict.keys()
    # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # r is for "solid red line"
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
    ]

    probability_model = keras.Sequential([
        model, 
        layers.Activation('sigmoid')
        ])
    
    probability_model.predict(examples)
    model.save('twitter_sentiment_model')