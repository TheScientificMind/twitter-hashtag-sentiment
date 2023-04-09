import re
from keras import layers

# https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

# (https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*

def preprocess_tweet(twt):
    pass

def vectorize_tweet(twts):
    # layer to convert words into ints
    vectorizer = layers.TextVectorization(
        max_tokens=2500,
        output_mode="int", 
        output_sequence_length=200,
        pad_to_max_tokens=True
    )

    vectorizer.adapt(twts) # converts train_text to ints

    return(twts)
