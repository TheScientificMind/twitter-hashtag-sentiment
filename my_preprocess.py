import re
import pandas as pd
import tensorflow as tf
from keras import layers
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from replace_dicts import abbrev, emoji_dict, emoticons

# commonly used source: https://www.kaggle.com/code/greyisbetter/twitter-sentiment-analysis-u-b-d-fs

class Preprocessor:
    def __init__(self, twt = "", twts=[]):
        self.twt = twt
        self.twts = twts
        self.stop_words = stopwords.words("english")
        self.lemma = WordNetLemmatizer()
        self.slang = abbrev
        self.emojis = emoji_dict
        self.emoticons = emoticons

    def replace_emojis(self, twt):
        for emoji, value in self.emojis.items():
            if emoji in twt:
                twt = twt.replace(emoji, f" {value} ")

        twt = twt.split()
        twt = " ".join(twt).strip()

        return twt

    def replace_emoticons(self, twt):
        for emoticon, value in self.emoticons.items():
            if emoticon in twt:
                twt = twt.replace(emoticon, f" {value} ")

        twt = twt.split()
        twt = " ".join(twt).strip()

        return twt

    def preprocess_twt(self, twt):
        # following line source: https://stackoverflow.com/questions/11331982/
        twt = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", "", twt) # removes urls
        twt = re.sub(r"@\w+", "", twt) # removes mentions
        twt = re.sub(r"#\w+", "", twt) # removes hashtags
        twt = re.sub(r"[^a-zA-Z0-9]", " ", twt) # removes non-alphanumeric chars

        twt = twt.lower()
        twt = word_tokenize(twt)

        twt = [token for token in twt if token not in self.stop_words] # remove stowords (e.g. a, an, the)

        twt = [self.lemma.lemmatize(token) for token in twt if token] # stems words (e.g. babbled -> babble)

        twt = " ".join(twt).strip()

        return twt
    
    def replace_slang(self, twt):
        twt = word_tokenize(twt)

        # Replace slang words with their full form
        for i, word in enumerate(twt):
            if word in self.slang:
                twt[i] = self.slang[word]

        twt = " ".join(twt).strip()
                
        return twt
    
    def clean_twts(self):
        self.twts = self.twts.apply(self.replace_emojis)
        self.twts = self.twts.apply(self.replace_emoticons)
        self.twts = self.twts.apply(self.preprocess_twt)
        self.twts = self.twts.apply(self.replace_slang)
        return self.twts
    
    def clean_twt(self):
        self.twt = self.replace_emojis(self.twt)
        self.twt = self.replace_emoticons(self.twt)
        self.twt = self.preprocess_twt(self.twt)
        self.twt = self.replace_slang(self.twt)
        return self.twt