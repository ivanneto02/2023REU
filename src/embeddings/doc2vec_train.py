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
    df = pd.read_csv(WIKI_DATA_PATH + WIKI_DATA_FILE, nrows=None)

    df["definition"] = df["name"] + " " + df["definition"]
    df.dropna(subset=["definition"], inplace=True)

    sentences = df["definition"].to_list()
    cuis      = df["cui"].to_list()

    print("     - Preprocessing sentences")
    sentences_pre = []
    for sen in tqdm.tqdm(sentences):
        sentences_pre.append(preprocess_string(remove_stopwords(sen.lower())))

    print("     - Tagging documents")
    documents = [ TaggedDocument(words=doc, tags=[cuis[i]]) for i, doc in enumerate(sentences_pre) ]

    modeld = Doc2Vec(
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

    # Build the vocabulary from the documents
    print("     - Building vocabulary")
    modeld.build_vocab(tqdm.tqdm(documents, total=len(documents)))

    print("     - Training model")
    for epoch in range(EPOCHS):
        # Wrap the documents iterator with tqdm and set leave=False to overwrite the progress bar
        documents_with_progress = tqdm.tqdm(documents, total=len(documents), desc="Epoch {:02d}/{:02d}".format(epoch+1, EPOCHS))
        
        # Train the model for the current epoch
        modeld.train(documents_with_progress, total_examples=modeld.corpus_count, epochs=1)

    print("     - Saving model")
    modeld.save(MODEL_SAVE_PATH + MODEL_DOC2VEC)