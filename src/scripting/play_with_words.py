import word2vec
import pandas as pd

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from keras import layers

def main():
    df = pd.read_csv("/mnt/c/Users/ivana/Desktop/Documents/Research/UCR/DS-PATH/working_dir/data/ready/ready_to_embed.csv", nrows=None)

    sentence = df["definition"].iloc[0]
    tokens   = list(sentence.lower().strip().split(" "))
    
    vocab, index = {}, 1 # start indexing from 1
    vocab['<pad>'] = 0   # padding token, not needed

    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1
    vocab_size = len(vocab)
    print(vocab)

    pass

if __name__ == "__main__":
    main()