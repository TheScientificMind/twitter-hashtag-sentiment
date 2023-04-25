import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import LSTM, Dropout, Embedding, Dense, GlobalMaxPool1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
from my_preprocess import preprocess_twts

"""
dataset: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

frequently used sources: 
https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
https://www.kaggle.com/code/irasalsabila/twitter-sentiment
"""

# layer to convert text to ints
vectorizer = layers.TextVectorization(
    max_tokens=2500,
    output_mode="int", 
    output_sequence_length=200,
    pad_to_max_tokens=True
)

# importing the dataset
df = pd.read_csv("tweets.csv")

# preprocessing
df["text"] = preprocess_twts(df["text"].astype(str))
df = df.dropna()
vectorizer.adapt(np.array(df["text"]))
x = vectorizer(np.array(df["text"]))
y = to_categorical(df["sentiment"].astype(int), num_classes=3)
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.1, random_state=42)

# building the model
model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(128, return_sequences=True),
    GlobalMaxPool1D(),
    Dense(36, activation="relu"),
    Dropout(.1),
    Dense(24, activation="relu"),
    Dropout(.1),
    Dense(16, activation="relu"),
    Dropout(.1),
    Dense(3, activation="softmax")
])

# saving vectorizer, source: stackoverflow.com/questions/65103526
pickle.dump({"config": vectorizer.get_config(),
            "weights": vectorizer.get_weights()}
            , open("tv_layer.pkl", "wb"))

opt = RMSprop(learning_rate=0.0012, rho=0.7, momentum=0.5)

# compiling the model
model.compile(
    optimizer=opt, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
) 

early_stop = EarlyStopping(monitor="val_accuracy", patience=3)

# training the model
history = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    batch_size=100,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
    callbacks=early_stop
)

# get information for graphing
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

# graphs loss over epochs
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.png")
plt.show()
plt.clf()

# graphs accuracy over epochs
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "r", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")
plt.show()
plt.clf()

accuracy = model.evaluate(model, x_test, y_test)

model.save("twitter_model")