import re
import pandas as pd
import tensorflow as tf
from keras import layers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# following four lines source: https://stackoverflow.com/questions/65103526/
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
vectorizer = layers.TextVectorization.from_config(from_disk['config'])
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk['weights']) 

def preprocess_twt(twt):
    # following line: https://stackoverflow.com/questions/11331982
    twt = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", "", twt) # removes urls
    twt = re.sub(r"@\w+", "", twt) # removes mentions
    twt = re.sub(r"#/w+", "", twt) # removes hashtags
    twt = re.sub(r"[^a-zA-Z0-9]", " ", twt) # removes non-alphanumeric chars

    twt = twt.lower()

    twt = word_tokenize(twt)

    stop_words = stopwords.words('english')
    twt = [token for token in twt if token not in stop_words] # remove stowords (e.g. a, an, the)

    stemmer = PorterStemmer()
    twt = [stemmer.stem(token) for token in twt] # stems words (e.g. babbled -> babble)

    twt = ' '.join(twt)

    return twt

def preprocess_twts(twts):
    twts.adapt(preprocess_twt)

# load vectorizer and vectorize data
def vectorize_twts(twts):
    vectorizer.adapt(twts) # converts train_text to ints
    return(twts)