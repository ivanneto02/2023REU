import pandas as pd
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm
import numpy as np

from bs4 import BeautifulSoup, SoupStrainer

def main():
    df = pd.read_csv(
            os.environ["_SCRAPED_DATA_PATH"] + os.environ["_SCRAPED_DATA_FILE"],
            nrows=1
    )

    strainer = SoupStrainer("p")

    soup = BeautifulSoup(df["raw_html"].iloc[0], parse_only=strainer, features="lxml")

    print(str(soup))

if __name__ == "__main__":
    main()