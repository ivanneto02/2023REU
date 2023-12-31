import pandas as pd
from config import *
from .mysql_connect import connect
from .mysql_disconnect import disconnect
import tqdm
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

    print(f"   - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)

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
     	        DISTINCT COALESCE(MRDEF.CUI, MRCONSO.CUI), MRCONSO.STR, MRDEF.DEF
            FROM
                MRCONSO
            LEFT OUTER JOIN
                MRDEF
            ON
                MRCONSO.`CUI` = MRDEF.`CUI`             
        '''

        # DataFrame storing
        # cui, str, def
        umls_df = None
        columns = ["cui", "name", "umls_definition"]

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

            print("     - Dropping duplicates")
            umls_df.drop_duplicates(subset=["cui", "umls_definition"], ignore_index=True, inplace=True)

        except Exception as e:
            print("         ! Executing load UMLS query went wrong !")
            print(e)

        print(f"         - Length of data: {len(umls_df)}")

        # print(umls_df.head(10))

        return umls_df

        # Make sure to disconnect
    except Exception as e:
        print("        ! Loading UMLS went wrong !")
        print(e)

def load_nonumls():
    df = pd.read_csv(SCRAPED_DATA_PATH + SCRAPED_DATA_FILE, nrows=NROWS)

    print(f"        - Length of data: {len(df)}")
    # filter into desired columns
    df = df[["name", "raw_html"]]

    df["source"] = "scraped"
    df.rename(inplace=True, columns={'raw_html' : 'scraped_definition'})
    return df