import pickle
from keras import layers

# layer to convert words into ints
vectorizer = layers.TextVectorization(
    max_tokens=2500,
    output_mode="int", 
    output_sequence_length=200,
    pad_to_max_tokens=True
)

# Pickle the config and weights
pickle.dump({
    "config": vectorizer.get_config(),
    "weights": vectorizer.get_weights()}, 
    open("tv_layer.pkl", "wb")
)