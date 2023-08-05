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

async def fetch_multiple(names, api):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, name, api) for name in names]
        responses = await asyncio.gather(*tasks)
    return responses

pd.options.display.max_rows = 1000

async def scrape():
    print("     - Loading Data")
    df = pd.read_csv(READY_DATA_PATH + READY_DATA_FILE, nrows=None)

    print(df.head(100))

    print(f"     - Length of data: {len(df)}")

    mediawikiapi = MediaWikiAPI()

    # Scraping
    parallel_size = 50

    if (len(df) < PARALLEL_SIZE):
        parallel_size = len(df)

    responses_df = pd.DataFrame()

    print("     - Starting Scraping Process")
    for i in tqdm.tqdm(range(0, len(df)//parallel_size + 1), desc="     Parallel Scraping"):
        time.sleep(1)
        try:
            terms = df["name"].iloc[i*parallel_size:(i*parallel_size)+parallel_size].to_list()

            # # Get term name
            # name = df["name"].iloc[i - 1] # Name to search
            # language_code = 'en'
            # number_of_results = 1
            # headers = {
            #     "User-Agent" : "TestingAPI (ineto001@ucr.edu)"
            # }
            # base_url = "https://api.wikimedia.org/core/v1/wikipedia/"
            # endpoint = "/search/page"
            # url = base_url + language_code + endpoint
            # parameters = {
            #     "q" : name,
            #     "limit" : number_of_results
            # }
            # response = requests.get(url, headers=headers, params=parameters)
            # json_response = json.loads(response.text)
            # # Get the first page in the result
            # page = json_response["pages"][0]
            # article_url   = "https://" + language_code + ".wikipedia.org/wiki/" + page["key"]
            # article = requests.get(article_url).content
            # df["definition"] = df["definition"] + " " + process_html(article)

            # searchlist = mediawikiapi.search(name, results=1)
            # firstterm = searchlist[0] # get the first term
            # page = mediawikiapi.page(firstterm)
            # content = page.url

            responses = await fetch_multiple(terms, mediawikiapi)
            
            # Limit words in each responses text
            for j in range(0, len(responses)):
                if responses[j] == "":
                    continue
                responses[j] = " ".join(process_html(responses[j]).split(" ")[:KEEPWORDS])

            # print(terms)
            # print(responses)

            responses_df = pd.concat([ responses_df, pd.DataFrame(responses) ], ignore_index=True)

            # print(responses_df)

            # print( " ".join(content.split(" ")[:KEEPWORDS][:100] ))
            # Append definition to the current definition and keep only certain amount of words
            # df["definition"] = df["definition"] + " " + " ".join(content.split(" ")[:KEEPWORDS])

        except Exception as e:
            print(e)

    # print(len(responses_df))
    # print(len(df))

    # print(responses_df)
    # print(df)

    df["definition"] = df["definition"] + " " + responses_df[0]

    # print(df)

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