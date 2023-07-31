import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.test.utils import common_texts

from config import *
import tqdm
import numpy as np
from multiprocessing import Pool

def doc2vec_train():
    print(" > Doc2Vec training...")
    print("     - Reading data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=1000)

    sentences = df["definition"].to_list()
    cuis      = df["cui"].to_list()

    modelw = Word2Vec.load(MODEL_SAVE_PATH + MODEL_WORD2VEC)

    print("     - Preprocessing sentences")
    # sentences = df["definition"].to_list()
    sentences_pre = []
    for sen in tqdm.tqdm(sentences):
        sentences_pre.append(preprocess_string(remove_stopwords(sen.lower())))

    # print("     - Pre-embedding sentences")
    # embeds = []
    # for sent in tqdm.tqdm(sentences_pre):
    #     embeds.append([ modelw.wv[i] for i in sent ])

    # print(embeds[0])

    print("     ")
    documents = [ TaggedDocument(words=doc, tags=[cuis[i]]) for i, doc in enumerate(sentences_pre) ]

    print(documents[0])

    # modelw = Word2Vec.load(MODEL_SAVE_PATH + MODEL_WORD2VEC)

    # print("     - Embedding sentences")
    # embedded = []
    # for sen in tqdm.tqdm(sentences_pre):
    #     embedded.append(modelw.wv[sen])

    print("     - Training Doc2Vec")
    modeld = Doc2Vec(
        tqdm.tqdm(documents),
        vector_size = LAYER_SIZE,
        window = WINDOW_SIZE,
        min_count = MIN_COUNT,
        workers = WORKERS,
        epochs = EPOCHS,
        negative = NEGATIVE_SAMPLES,
        alpha = LEARNING_RATE,
        ns_exponent = NS_EXPONENT,
        dm = DM,
        hs = HS,
        dm_concat = DM_CONCAT,
        dm_mean   = DM_MEAN
    )

    # inferred_vector = modeld.infer_vector(["test", "ting"])
    # print(inferred_vector)

    modeld.save(MODEL_SAVE_PATH + MODEL_DOC2VEC)