import pandas as pd
from config import *

def boostconcepts():
    print(" > Boosting concepts")

    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    print(f"     - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df.to_csv(READY_DATA_PATH + READY_DATA_FILE, index=False)