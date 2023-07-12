import pandas as pd
from config import *
from pymetamap import MetaMap
import time

def extract_nonumls_definitions():
    print(" > Extracting non-UMLS definitions... (may take a while)")
    
    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=NROWS)
    print(df.head(5))

    print("     - Extracting definitions")
    metamap_base_dir = '/home/ivan/metamapdownload/public_mm/'
    metamap_bin_dir = 'bin/metamap'

    metamap_pos_server_dir = 'bin/skrmedpostctl'
    metamap_wsd_server_dir = 'bin/wsdserverctl'

    print("     - Starting `skrmedpostctl` and `wsdserverctl`")
    os.system(metamap_base_dir + metamap_pos_server_dir + ' start') # Part of speech tagger
    os.system(metamap_base_dir + metamap_wsd_server_dir + ' start') # Word sense disambiguation 

    print("     - Waiting for `skrmedpostctl` and `wsdserverctl` to start")
    time.sleep(10)

    metam = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)
    term_list = df["name"].to_list()
    term_indexes = list(range(len(term_list)))

    concepts, errs = metam.extract_concepts(term_list, term_indexes)

    # Look at the output:
    for i in range(0, 5):
        print(concepts[i])

    print("     - Stopping `skrmedpostctl` and `wsdserverctl`")
    os.system(metamap_base_dir + metamap_pos_server_dir + ' stop') # Part of speech tagger
    os.system(metamap_base_dir + metamap_wsd_server_dir + ' stop') # Word sense disambiguation 

    print("     - Waiting for `skrmedpostctl` and `wsdserverctl` to stop")
    time.sleep(5)

    print("     - Done")