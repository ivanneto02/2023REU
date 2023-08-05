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

    df = pd.read_csv(os.environ["_READY_DATA_PATH"] + os.environ["_READY_DATA_FILE"], nrows=10)

    name_test = df["name"].iloc[0] # Name to search
    # name_test = "UK"

    language_code = 'en'
    number_of_results = 1
    headers = {
        "User-Agent" : "TestingAPI (ineto001@ucr.edu)"
    }

    base_url = "https://api.wikimedia.org/core/v1/wikipedia/"
    endpoint = "/search/page"
    url = base_url + language_code + endpoint
    
    parameters = {
        "q" : name_test,
        "limit" : number_of_results
    }

    response = requests.get(url, headers=headers, params=parameters)
    json_response = json.loads(response.text)

    print(f"Searching for {name_test}")

    # Get the first page in the result
    page = json_response["pages"][0]

    display_title = page["title"]
    article_url   = "https://" + language_code + ".wikipedia.org/wiki/" + page["key"]

    # Description
    try:
        article_description = page["description"]
    except:
        article_description = "NA"

    # Thumbnail
    try:
        thumbnail_url = "https:" + page["thumbnail"]["url"]
    except:
        thumbnail_url = "NA"

    print(f"Title: {display_title}")
    print(f"URL: {article_url}")
    print(f"Description: {article_description}")
    print(f"Thumbnail: {thumbnail_url}")

if __name__ == "__main__":
    main()