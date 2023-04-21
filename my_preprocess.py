import re
import pandas as pd
import tensorflow as tf
from keras import layers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# layer to convert words into ints
vectorizer = layers.TextVectorization(
    max_tokens=2500,
    output_mode="int", 
    output_sequence_length=200,
    pad_to_max_tokens=True
)

def preprocess_twt(twt):
    # following line: https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners
    twt = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", "", twt) # removes urls
    twt = re.sub(r"@\w+", "", twt) # removes mentions
    twt = re.sub(r"#/w+", "", twt) # removes hashtags
    twt = re.sub(r"[^a-zA-Z0-9]", " ", twt) # removes non-alphanumeric chars

    twt = twt.lower()

    twt = word_tokenize(twt)

    stop_words = stopwords.words("english")
    twt = [token for token in twt if token not in stop_words] # remove stowords (e.g. a, an, the)

    stemmer = PorterStemmer()
    twt = [stemmer.stem(token) for token in twt] # stems words (e.g. babbled -> babble)

    twt = " ".join(twt).strip()

    return twt

def preprocess_twts(twts):
    twts.apply(preprocess_twt)
    return twts

# load vectorizer and vectorize data
def vectorize_twts(twts):
    vectorizer.adapt(twts) # converts train_text to ints
    twts = vectorizer(twts)
    return twts