import pandas as pd
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm
import numpy as np

def main():
    print(" > Reading data")
    df = pd.read_csv(os.environ["_READY_DATA_PATH"] + os.environ["_EMBED_DEF_FILE"], nrows=None)
    
    print(" > Grabbing data")
    documents = df["definition"].to_numpy()
    documents = [ np.array(i) for i in documents ]
    
    print(documents)

    print(" > Training Doc2Vec")
    model = Doc2Vec(
        tqdm.tqdm(documents),
        window = 7,
        min_count = 1,
        workers = 4
    )

    inferred_vector = model.infer_vector(["test", "ting"])

if __name__ == "__main__":
    main()