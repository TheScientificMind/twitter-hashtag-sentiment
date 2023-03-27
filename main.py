import sys
import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# import spacy
# import re
# import tweepy

# good resource: https://www.youtube.com/watch?v=lKoPIQdRBME&ab_channel=MachineLearningwithPhil

# ensures the file isn't being imported as a module
if __name__ == "__main__":
    try:
        df = pd.read_csv("tweetsdataset.csv")
        
        # lines 16-20 from https://stackoverflow.com/questions/43697240
        rng = RandomState()
        train = df.sample(frac=0.9, random_state=rng)
        test = df.loc[~df.index.isin(train.index)]

        hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

        # invalid hashtag causes program to stop
        if not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
            print("Your hashtag was invalid. Please run the program and try again.")
            sys.exit()
    except Exception as err:
        print(f"The program encountered the error:\n\n{err}\n\nPlease try again.")
        
