import requests
from config import *
import sys
import pandas as pd
import json
import time
from bs4 import BeautifulSoup, SoupStrainer
import tqdm
import re
from mediawikiapi import MediaWikiAPI
import numpy as np
import aiohttp
import asyncio

# Basically just get rid of everything
# but all the paragraphs
strainer = SoupStrainer("p")
def process_html(x):
    try:
        soup = BeautifulSoup(x, parse_only=strainer, features="lxml")
        x = soup.text.strip()
        x = re.sub("\\n", "", x)
        return x
    except Exception as e:
        print(" - " + str(e))
        print(x)
        return ""

# Based on a term name (i.e. "headache"), we fetch the page
# that results from the first search result when searching
# wikipedia for that term name.
async def fetch(session, name, api):
    try:
        searchlist = api.search(name, results=1)
        firstterm = searchlist[0]
        page = api.page(firstterm)
        url = page.url
    except:
        return ""
    async with session.get(url) as response:
        return await response.text()

# Based on multiple term names, run fetch for each one of
# those. Fetch is explained in its above comment.
async def fetch_multiple(names, api):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, name, api) for name in names]
        responses = await asyncio.gather(*tasks)
    return responses

# For visual purposes if you need
# pd.options.display.max_rows = 1000

async def scrape():
    print("     - Loading Data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=None)

    df["wiki_definition"] = ""

    print(f"     - Length of data: {len(df)}")

    mediawikiapi = MediaWikiAPI()

    # Scraping
    parallel_size = 500

    if (len(df) < PARALLEL_SIZE):
        parallel_size = len(df)

    responses_df = pd.DataFrame()

    # Basically this takes every single row in READY_DATA_PATH + READY_DATA_FILE, and finds a
    # wikipedia search result to grab that definition and add to the current definition. Important
    # because it will play a huge role in having additional definition information for training the
    # doc2vec model.
    print("     - Starting Scraping Process")
    for i in tqdm.tqdm(range(0, len(df)//parallel_size + 1), desc="     Parallel scraping"):
        time.sleep(1)
        try:
            terms = df["name"].iloc[i*parallel_size:(i*parallel_size)+parallel_size].to_list()

            responses = await fetch_multiple(terms, mediawikiapi)
            # Limit words in each responses text
            for j in range(0, len(responses)):
                if responses[j] == "":
                    continue
                responses[j] = process_html(responses[j])
            responses_df = pd.concat([ responses_df, pd.DataFrame(responses) ], ignore_index=True)

        except Exception as e:
            responses = [""]*parallel_size
            responses_df = pd.concat([ responses_df, pd.DataFrame(responses) ], ignore_index=True)
            print(e)

    # Concatenate the wikipedia definition to the current definition
    df["wiki_definition"] = df["wiki_definition"] + " " + responses_df[0]

    # The little wrapper ensures we don't have anything open
    # before we save because this step takes so long.
    done = False
    while not done:
        try:
            print(f"     - Saving to {WIKI_DATA_PATH + WIKI_DATA_FILE}")
            df.to_csv(WIKI_DATA_PATH + WIKI_DATA_FILE, index=False)
            done = True
        except:
            a = input("     !! Please close the instance of the Wikipedia data. Press 'Enter' when ready to save.")

def scrapeterms():
    try:
        if (sys.argv[1].strip() == 'scraping' and sys.argv[2].strip() == '1'):

            # Run asynchronous scraping
            loop = asyncio.get_event_loop()
            loop.run_until_complete(scrape())

        else:
            print("         ** usage: python ./start_scrapingwiki.py scraping 1")

    except Exception as e:
        print("         - Skipping Wikipedia Scraping Module")
        print('''       ** You may not run this module yet because you did NOT set the variable "python ./start_pipeline.py scraping 1"''')
        print('''           ** This part of the module takes an incredibly long time to run. If not ran carefully,''')
        print('''              you will lose your Wikipedia data and spend many hours scraping new data.''')
        print(e)

if __name__ == "__main__":
    scrapeterms()