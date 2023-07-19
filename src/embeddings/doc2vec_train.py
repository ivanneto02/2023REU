import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.test.utils import common_texts

from config import *
import tqdm
import numpy as np

def doc2vec_train():
    print(" > Doc2Vec training...")
    print("     - Reading data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=1000)

    sentences = df["definition"].to_list()

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

    documents = [ TaggedDocument(doc, [i]) for i, doc in enumerate(sentences_pre) ]

    # modelw = Word2Vec.load(MODEL_SAVE_PATH + MODEL_WORD2VEC)

    # print("     - Embedding sentences")
    # embedded = []
    # for sen in tqdm.tqdm(sentences_pre):
    #     embedded.append(modelw.wv[sen])

    print("     - Training Doc2Vec")
    modeld = Doc2Vec(
        tqdm.tqdm(documents),
        vector_size = 200,
        window = 7,
        min_count = 1,
        workers = 4,
        epochs = 20,
        negative = 4
    )

    # inferred_vector = modeld.infer_vector(["test", "ting"])
    # print(inferred_vector)

    modeld.save(MODEL_SAVE_PATH + MODEL_DOC2VEC)