import sys
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import spacy
# import re
# import tweepy

# ensures the file isn't being imported as a module
if __name__ == "__main__":
    hashtag = input("What hashtag would you like to analyze (e.g. #photography): ").lower().strip()

    # invalid hashtag causes program to stop
    if not hashtag[1:].isalnum() or hashtag[1:].isnumeric() or "#" not in hashtag:
        print("Your hashtag was invalid. Please run the program and try again.")
        sys.exit()
        
