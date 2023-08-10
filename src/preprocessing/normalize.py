from config import *
import time
from pymetamap import MetaMap
import pandas as pd
import subprocess

import io
import sys

def normalize():
    print(" > Normalizing...")
    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    print("     - NO normalizing step yet")
    print(f"     - Saving to {READY_DATA_PATH + READY_DATA_FILE}")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)