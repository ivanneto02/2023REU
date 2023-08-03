import pandas as pd
from config import *

def drop_nas():
    print(" > Dropping NAs...")

    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    print("     - Removing NAN values")
    df.dropna(inplace=True)

    print(f"     - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=True)

    pass