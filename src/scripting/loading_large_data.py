import pandas as pd
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

from bs4 import BeautifulSoup, SoupStrainer

import tqdm

from ParallelPyMetaMap import ppmm

def process_chunk(chunk):
    strainer = SoupStrainer("p")
    for i in range(0, len(chunk)):
        soup = BeautifulSoup(chunk["raw_html"].iloc[i], parse_only=strainer, features="lxml")
        chunk["raw_html"].iloc[i] = str(soup.text)
    return chunk

def main():

    df = pd.DataFrame()
    i = 1
    for chunk in pd.read_csv(
            os.environ["_SCRAPED_DATA_PATH"] + os.environ["_SCRAPED_DATA_FILE"],
            chunksize = int(os.environ["_CHUNKSIZE"])
    ):
        # process chunk with beautifulsoup to only keep the needed parts (<p> fields)
        print(f"    - Loading chunk ({i})")
        df = pd.concat([df, process_chunk(chunk)])
        i += 1

    print(f" > Length of data: {len(df)}")
    # filter into desired columns
    df = df[["name", "raw_html"]]

    df["source"] = "scraped"
    df.rename(inplace=True, columns={'raw_html' : 'definition'})

    print("> Printing")
    print(df.head(5))

if __name__ == "__main__":
    main()