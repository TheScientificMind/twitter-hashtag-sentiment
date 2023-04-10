import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime
from datasets import load_dataset
from my_preprocess import preprocess_twts, vectorize_twts

# frequently used source: 
# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
# https://www.youtube.com/watch?v=hprBCp_UJN0&ab_channel=CodeHeroku

tweets = load_dataset("carblacac/twitter-sentiment-analysis", "None")

split_point = round(len(tweets) * 0.95) # the id at which the tweet dataset should be split

train = tweets[:split_point]
train_text = preprocess_twts(train["text"].astype(str))
train_labels = train["label"].astype(int)

test = tweets[split_point:]
test_text = preprocess_twts(test["text"].astype(str))
test_labels = test["label"].astype(int)

vectorize_twts(train_text)
vectorize_twts(test_text)

# building the model
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=200),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16 ,activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# compiling the model
model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
) 

# training the model
history = model.fit(
    train_text, 
    train_labels, 
    epochs=25, 
    batch_size=512, 
    validation_data=(test_text, test_labels)
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# graph of accuracy and loss over epochs

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.savefig("loss.png")

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
plt.savefig("accuracy.png")

model.save("twitter_model") # saves the model