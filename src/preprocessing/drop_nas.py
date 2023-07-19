import pandas as pd
from config import *

def drop_nas():
    print(" > Dropping NAs...")

    print("     - Reading data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=NROWS)

    print("     - Removing NAN values")
    df.dropna(inplace=True)

    print("     - Saving")
    df.to_csv(READY_DATA_PATH + READY_DATA_FILE, index=True)

    pass