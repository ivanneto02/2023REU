from config import *
import pandas as pd
import numpy as np
import tqdm

def limit_def(x):
    try:
        x = x.strip()
        x = x.split(" ")
        x = x[:KEEPWORDS]
        x = " ".join(x)
    except Exception as e:
        x = np.nan
    return x

def limit_definitions():
    print(" > Limiting definitions...")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    print(f"    - Length of data {len(df)}")

    print("     - Removing excess words")

    tqdm.tqdm.pandas()
    df["umls_definition"]    = df["umls_definition"].progress_apply(limit_def)
    df["scraped_definition"] = df["scraped_definition"].progress_apply(limit_def)

    print(f"     - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)