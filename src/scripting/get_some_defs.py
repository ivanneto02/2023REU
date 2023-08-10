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
            os.environ["_READY_DATA_PATH"] + os.environ["_READY_DATA_FILE"],
            nrows=10
        )
    
    print(df.head(5))
    print(f"Length: {len(df)}")
    print(f"Columns: {df.columns}")

    df.to_csv(os.environ["_READY_DATA_PATH"] + "test.csv", index=False)

if __name__ == "__main__":
    main()