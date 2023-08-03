import pandas as pd
from config import *
from pymetamap import MetaMap
import time
from bs4 import BeautifulSoup, SoupStrainer
import re
from tqdm import tqdm

def extract_nonumls_definitions():
    print(" > Extracting NON-UMLS definitions")

    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    # separate into umls vs non-umls
    print("     - Separating scraped vs. umls")
    df_scraped = df.loc[df["source"] == "scraped"].copy()
    df_umls    = df.loc[df["source"] == "umls"].copy()

    try:
        # extract definitions from html
        tqdm.pandas()
        print("     - Extracting definitions (may take a while)")
        df_scraped["definition"] = df_scraped["definition"].progress_apply(extract)
    except Exception as e:
        print("             ! Something went wrong with extracting definitions !")
        print(e)

    # put the two together again
    df_final = pd.concat([df_scraped, df_umls], axis=0)

    # print(df_final.head(10))

    # save data
    print(f"     - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df_final.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)

strainer = SoupStrainer("p")
def extract(x):
    try:
        soup = BeautifulSoup(x, parse_only=strainer, parser="lxml", features="lxml")
        text = soup.get_text().replace("\n", "")
        text = text.replace("\t", " ")
        text = re.sub("\ +", " ", text)
        return text
    except Exception as e:
        return None