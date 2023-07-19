from config import *
import pandas as pd

def limit_def(x):
    x = x.strip()
    x = x.split(" ")
    x = x[:KEEPWORDS]
    x = " ".join(x)
    return x

def limit_definitions():
    print(" > Limiting definitions...")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=NROWS)

    print(f" > Length of data {len(df)}")
    
    print(" > Removing excess words")
    df["definition"] = df["definition"].apply(limit_def)

    print(" > Saving definitions...")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)