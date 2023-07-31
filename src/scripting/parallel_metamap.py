import pandas as pd
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm

from ParallelPyMetaMap import ppmm

def main():
    print(" > Reading data")

    df = pd.read_csv(
        os.environ["_READY_DATA_PATH"] + os.environ["_READY_DATA_FILE"],
        nrows=10
    )

    terms = df["definition"].to_list()

    test = ppmm(4, 
        "/home/ivan/metamapdownload/public_mm/bin/metamap",
        terms,
        "definition",
        machine_output=True
    )

    print(test)

if __name__ == "__main__":
    main()