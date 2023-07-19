import pandas as pd
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm

def main():
    print(" > Reading data")

    df = pd.read_csv(
            os.environ["_SCRAPED_DATA_PATH"] + os.environ["_SCRAPED_DATA_FILE"],
            nrows=10
        )
    
    print(df[["name", "raw_html", "source_name", "concept_type"]].head(5))
    print(f"Length: {len(df)}")
    print(f"Columns: {df.columns}")

if __name__ == "__main__":
    main()