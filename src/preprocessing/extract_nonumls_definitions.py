import pandas as pd
from config import *
from pymetamap import MetaMap
import time
from bs4 import BeautifulSoup, SoupStrainer
import re

def extract_nonumls_definitions():
    print(" > Extracting NON-UMLS definitions")

    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    # separate into umls vs non-umls
    print("     - Separating scraped vs. umls")
    df_scraped = df[df["source"] == "scraped"]
    df_umls    = df[df["source"] == "umls"]

    try:
        # extract definitions from html
        print("     - Extracting definitions (may take a while)")
        df_scraped["definition"] = df_scraped["definition"].apply(extract)
    except Exception as e:
        print("             ! Something went wrong with extracting definitions !")
        print(e)

    # put the two together again
    df_final = pd.concat([df_scraped, df_umls])
    print(df_final)

    # save data
    print("     - Saving")
    df_final.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)

def extract(x):
    strainer = SoupStrainer("p")
    soup = BeautifulSoup(x, parse_only=strainer)
    text = soup.get_text().replace("\n", "")
    text = text.replace("\t", " ")
    text = re.sub("\ +", " ", text)
    return text