import wikipediaapi

import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

import tqdm
import numpy as np

from bs4 import BeautifulSoup, SoupStrainer

import json
import requests

import time

def main():

    df = pd.read_parquet(
        "/mnt/d/Documents/Research/DSPATH/data/ready/Extended_CUI_Definitions_pt0.parquet",
        engine="pyarrow")

    print(df.head(10))
    print(df["extended_definition"].iloc[2])


if __name__ == "__main__":
    main()