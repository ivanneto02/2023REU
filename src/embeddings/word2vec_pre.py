import pandas as pd
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
import tqdm
from config import *
import numpy as np

def word2vec_pre():
    print(" > Word2Vec preprocessing...")

    print("     - Reading data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=1000)

    print("     - Preprocessing sentences")
    sentences = df["definition"].to_list()

    print(sentences[0][:100])

    sentences_pre = []
    for sen in tqdm.tqdm(sentences):
        sentences_pre.append(preprocess_string(remove_stopwords(sen.lower())))

    print(sentences_pre[0][:10])

    print("     - Compiling word2vec")
    model = Word2Vec(
        sentences = tqdm.tqdm(sentences_pre),
        vector_size = 200,
        window = 7,
        min_count = 1,
        workers = 4
    )

    print("     - Training word2vec")
    # train the models
    model.train(
        tqdm.tqdm(sentences_pre),
        total_examples=len(sentences_pre),
        epochs = 20
    )

    print("     - Saving")
    model.save(MODEL_SAVE_PATH + MODEL_WORD2VEC)