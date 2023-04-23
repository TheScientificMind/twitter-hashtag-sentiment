import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Embedding, Dense, GlobalMaxPool1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime
from sklearn.utils.class_weight import compute_class_weight
from my_preprocess import preprocess_twts, vectorize_twts

# frequently used source: 
# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/
# https://www.youtube.com/watch?v=hprBCp_UJN0&ab_channel=CodeHeroku
# https://www.kaggle.com/code/malekmavetgaming/classifying-tweets-using-nlp-in-tensorflow-2-0

# Source of the next 3 lines: https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners
columns  = ["id", "entity", "sentiment", "content"]
twt_encoding = "ISO-8859-1"
train = pd.read_csv("twitter_training.csv", encoding=twt_encoding, names=columns) # importing the dataset
tweets_test = pd.read_csv("twitter_validation.csv", encoding=twt_encoding, names=columns)

train["content"] = preprocess_twts(train["content"].astype(str))
train = train.dropna()
x_train = vectorize_twts(np.array(train["content"]))
y_train = train["sentiment"].astype(int)

weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train) # finding class weights for fixing dataset imbalance
weight = {i: weight[i] for i in range(len(weight))}

y_train = to_categorical(y_train, num_classes=3)

tweets_test["content"] = preprocess_twts(tweets_test["content"].astype(str))
tweets_test = tweets_test.dropna()
x_test = vectorize_twts(np.array(tweets_test["content"]))
y_test = to_categorical(tweets_test["sentiment"].astype(int), num_classes=3)

# building the model
model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(128, return_sequences=True),
    GlobalMaxPool1D(),
    Dense(64, activation = "relu"),
    Dropout(.2),
    Dense(16, activation = "relu"),
    Dropout(.2),
    Dense(3, activation = "softmax")
])

opt = RMSprop(learning_rate=0.03, rho=0.7, momentum=0.5)

# compiling the model
model.compile(
    optimizer=opt, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
) 

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 3)

# training the model
history = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    batch_size=100,
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=1,
    callbacks=early_stop,
    class_weight=weight
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
plt.show()
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
plt.clf()

model.save("twitter_model") # saves the model