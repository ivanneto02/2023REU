from config import *
import time
from pymetamap import MetaMap
import pandas as pd
import subprocess

import io
import sys
import tqdm

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import sys

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def tag_terms():
    print(" > Tagging terms...")
    
    print("     - Reading data")
    df = pd.read_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, nrows=None)

    # print(df.loc[df.isnull().any(axis=1)])

    # Separate UMLS vs non-UMLS in order to tag the non-UMLS
    # terms and then place them back together
    df_umls     = df.loc[~df["cui"].isna()] # UMLS terms
    # print(df_umls)

    df_umls.drop(columns=["index"])
    df_non_umls = df.loc[df["cui"].isna()]  # NON-UMLS terms
    # print(df_non_umls)

    # Drop CUI column, to be able to join with output table
    df_non_umls = df_non_umls.drop(columns=["cui"])

    # print(df_non_umls.head(5))
    # print(len(df_non_umls))

    metamap_base_dir = '/home/ivan/metamapdownload/public_mm/'
    metamap_bin_dir = 'bin/metamap'

    metamap_pos_server_dir = 'bin/skrmedpostctl'
    metamap_wsd_server_dir = 'bin/wsdserverctl'

    print("     - Starting `skrmedpostctl` and `wsdserverctl`")
    call1 = [metamap_base_dir + metamap_pos_server_dir, "start"]
    call2 = [metamap_base_dir + metamap_wsd_server_dir, "start"]

    with open(os.devnull, 'w') as fnull:
        subprocess.Popen(call1, stdout=fnull, stderr=subprocess.PIPE)
    with open(os.devnull, 'w') as fnull:
        subprocess.Popen(call2, stdout=fnull, stderr=subprocess.PIPE)
        
    print("     - Waiting for `skrmedpostctl` and `wsdserverctl` to start")
    time.sleep(5)

    metam = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)
    term_list = df_non_umls["name"].to_list()
    term_indexes = list(range(len(term_list)))

    # Only 10000 terms at a time for memory concerns.
    batchsize = 10000
    num_batches = (len(term_list) // batchsize) + 1
    remainder_terms = len(term_list) % batchsize

    print("     - Extracting CUIs from terms")

    with suppress_stdout_stderr():
        concepts = []
        for i in range(0, num_batches):
            if (i == num_batches - 1):
                print(f"         - Batch {i + 1}/{num_batches} ({remainder_terms}) [{(num_batches-1)*batchsize}:{''}]")
                curr_con, curr_err = metam.extract_concepts(
                    term_list[(num_batches-1)*batchsize:], term_indexes[(num_batches-1)*batchsize:]
                )
                concepts += curr_con
            else:
                print(f"         - Batch {i + 1}/{num_batches} ({batchsize}) [{i*batchsize}:{(i+1)*batchsize - 1}]")
                curr_con, curr_err = metam.extract_concepts(
                    term_list[i*batchsize:(i+1)*batchsize], term_indexes[i*batchsize:(i+1)*batchsize]
                )
                concepts += curr_con

    concepts_df = pd.DataFrame(concepts)[["index", "cui"]]
    concepts_df["index"] = concepts_df["index"].astype("Int64")

    # print("CONCEPTS DF:")
    # print(concepts_df)

    joined_df = pd.merge(df_non_umls, concepts_df, on="index", how="outer")
    # print(joined_df)

    final_df = pd.concat([joined_df, df_umls], axis=0)
    final_df.drop(columns=["index"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    print(f"     - Saving to {SAVE_DATA_PATH + SAVE_DATA_FILE}")
    final_df.to_csv(SAVE_DATA_PATH + SAVE_DATA_FILE, index=False)

    print("     - Stopping `skrmedpostctl` and `wsdserverctl`")

    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    os.system(metamap_base_dir + metamap_pos_server_dir + ' stop') # Part of speech tagger
    os.system(metamap_base_dir + metamap_wsd_server_dir + ' stop') # Word sense disambiguation 
    sys.stdout = sys.__stdout__

    print("     - Waiting for `skrmedpostctl` and `wsdserverctl` to stop")
    time.sleep(5)

    print("     - Done")