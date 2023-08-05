import src.scrapingwiki as scrapingwiki

def main():
    '''
        Starts scraping wikipedia based on definitions gathered from public data AND UMLS. This will:
            (1) Iterate through every single row in the dataset, which contains unique CUIs
            (2) Scrape wikipedia for an article for each of these unique CUIs
                - 200 articles per second
            (3) Save the dataset with the Wikipedia definitions
    '''

    try:
        # ---------
        print("> Starting [SCRAPING] process")
        scrapingwiki.scrapeterms()
        print("> Done!")

    except Exception as e:
        print("   ! Something went wrong !")
        print(e)

if __name__ == "__main__":
    main()