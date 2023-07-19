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
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=NROWS)

    print(f" > Length of data {len(df)}")

    print(" > Removing excess words")

    tqdm.tqdm.pandas()
    df["definition"] = df["definition"].progress_apply(limit_def)

    print(" > Saving definitions...")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)