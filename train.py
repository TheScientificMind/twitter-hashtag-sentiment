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

# Source of the next 3 lines: https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners
columns  = ["id", "entity", "sentiment", "content"]
twt_encoding = "ISO-8859-1"
tweets_training = pd.read_csv("twitter_training.csv", encoding=twt_encoding, names=columns) # importing the dataset
tweets_test = pd.read_csv("twitter_validation.csv", encoding=twt_encoding, names=columns)

tweets_training["content"] = preprocess_twts(tweets_training["content"].astype(str))
tweets_training.dropna()
train_text = vectorize_twts(np.array(tweets_training["content"]))
train_labels = tweets_training["sentiment"].astype(int)

tweets_test["content"] = preprocess_twts(tweets_test["content"].astype(str))
tweets_test.dropna()
test_text = vectorize_twts(np.array(tweets_test["content"]))
test_labels = tweets_test["sentiment"].astype(int)

# building the model
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=200),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16 ,activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

# compiling the model
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
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

# graphs accuracy and loss over epochs

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("loss.png")
plt.show()
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.savefig("accuracy.png")
plt.show()
plt.clf()

model.save("twitter_model") # saves the model