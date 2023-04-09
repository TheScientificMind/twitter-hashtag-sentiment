import re
import pandas as pd
from keras import layers
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

# https://www.kirenz.com/post/2022-06-17-sentiment-analysis-with-tensorflow-and-keras/

def preprocess_twt(twt):
    # following line: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
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

def vectorize_twts(twts):
    # layer to convert words into ints
    vectorizer = layers.TextVectorization(
        max_tokens=2500,
        output_mode="int", 
        output_sequence_length=200,
        pad_to_max_tokens=True
    )

    vectorizer.adapt(twts) # converts train_text to ints

    return(twts)