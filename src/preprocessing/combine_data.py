import pandas as pd
from config import *
from .mysql_connect import connect
from .mysql_disconnect import disconnect

from bs4 import BeautifulSoup, SoupStrainer


def combine_data():
    print(" > Combining data...")
    print("   - Loading all non-UMLS data")
    df_nonumls = load_nonumls()

    print("   - Loading UMLS data")
    df_umls = load_umls()

    print("   - Creating DataFrame")
    df = pd.concat([df_nonumls, df_umls], axis=0)
    df.reset_index(inplace=True)

    # print(df)

    print(f"   - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)
    pass

def load_umls():
    try:
        # Connect to MYSQL
        connection = connect(
            host = UMLS_HOST,
            database = UMLS_DB,
            user = UMLS_USER,
            password = UMLS_PASSW
        )

        # Grab the table
        definitions_query = f'''
            SELECT                                              
                DISTINCT MRDEF.CUI, MRDEF.DEF     
            FROM                                                
                MRDEF, MRCONSO                                  
            WHERE                                               
                MRDEF.CUI = MRCONSO.CUI                         
        '''

        # DataFrame storing
        # cui, str, def
        umls_df = None
        columns = ["cui", "definition"]

        if NROWS is not None:
            definitions_query += f'''
                LIMIT {NROWS};       
            '''
        else:
            definitions_query += ';'
        
        try:
            cursor = connection.cursor()
            cursor.execute(definitions_query)
            results = cursor.fetchall()
            umls_df = pd.DataFrame(results, columns=columns)
            umls_df["source"] = "umls"

        except Exception as e:
            print("         ! Executing load UMLS query went wrong !")
            print(e)

        return umls_df

        # Make sure to disconnect
    except Exception as e:
        print("        ! Loading UMLS went wrong !")
        print(e)

def process_chunk(chunk):
    strainer = SoupStrainer("p")
    chunk[]

def load_nonumls():

    final_df = pd.DataFrame()

    for chunk in pd.read_csv(SCRAPED_DATA_PATH + SCRAPED_DATA_FILE, nrows=NROWS, chunksize=10000):
        # process chunk with beautifulsoup to only keep the needed parts (<p> fields)
        final_df = pd.concat([final_df, process_chunk(chunk)])

    print(f" > Length of data: {len(df)}")
    # filter into desired columns
    df = df[["name", "raw_html"]]
    df["source"] = "scraped"
    df.rename(inplace=True, columns={'raw_html' : 'definition'})
    return df