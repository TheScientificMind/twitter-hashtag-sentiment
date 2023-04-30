import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.layers import LSTM, Dropout, Embedding, Dense, GlobalMaxPool1D, BatchNormalization
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from my_preprocess import Preprocessor # importing my own preprocessing class

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
df.head(5)

# preprocessing
preprocessor = Preprocessor(twts=df["text"].astype(str)) # creates an instance of the Preprocessor class
df["text"] = preprocessor.clean_twts()
df = df.dropna()
vectorizer.adapt(np.array(df["text"]))
x = vectorizer(np.array(df["text"]))
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.1, random_state=42)

#  computing the sentiment weights
weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
weight = {i: weight[i] for i in range(len(weight))}
weight[1] = weight[1] - 0.2

# converting sentiments to categorical
y_train, y_test = to_categorical(y_train.astype(int), num_classes=3), to_categorical(y_test.astype(int), num_classes=3)

# build the model
# regularizer sourcee: https://medium.com/data-science-365/how-to-apply-l1-and-l2-regularization-techniques-to-keras-models-da6249d8a469
model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(128, return_sequences=True),
    GlobalMaxPool1D(),
    Dense(64, activation="relu", kernel_regularizer=regularizers.L1L2(l1=0.00001, l2=0.00001)),
    Dropout(.2),
    Dense(32, activation="relu", kernel_regularizer=regularizers.L1L2(l1=0.00001, l2=0.00001)),
    Dropout(.2),
    Dense(3, activation="softmax")
])

# saving vectorizer, source: stackoverflow.com/questions/65103526s
pickle.dump({"config": vectorizer.get_config(),
            "weights": vectorizer.get_weights()},
            open("tv_layer.pkl", "wb"))

# compiling the model
opt = RMSprop(learning_rate=0.0012, rho=0.7, momentum=0.5)
model.compile(
    optimizer=opt, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
) 

# stops training early if val accuracy starts to decay
early_stop = EarlyStopping(monitor="val_accuracy", patience=2)

# training the model
history = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    batch_size=100,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
    callbacks=early_stop,
    class_weight=weight
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
plt.show()
plt.clf()

# graphs accuracy over epochs
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "r", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
plt.clf()

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test) # ~70% accuracy
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# save the model
model.save("twitter_model")