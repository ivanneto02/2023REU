import pandas as pd
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm

def main():
    print(" > Reading data")
    df = pd.read_csv(os.environ["_READY_DATA_PATH"] + os.environ["_READY_DATA_FILE"], nrows=None)
    
    print(" > Preprocessing sentences")
    sentences = df["definition"].to_list()
    sentences_pre = []
    for sen in tqdm.tqdm(sentences):
        sentences_pre.append(preprocess_string(remove_stopwords(sen.lower())))

    print(" > Compiling word2vec")
    model = Word2Vec(
        sentences = tqdm.tqdm(sentences_pre),
        vector_size = 200,
        window = 7,
        min_count = 1,
        workers = 4
    )

    print(" > Training word2vec")
    # train the models
    model.train(
        tqdm.tqdm(sentences_pre),
        total_examples=len(sentences_pre),
        epochs = 20
    )

    print(" > Embedding sentences")
    embedded = []
    for sen in tqdm.tqdm(sentences_pre):
        embedded.append(model.wv[sen])

    for i in range(0, 5):
        print(embedded[i][:5])

if __name__ == "__main__":
    main()