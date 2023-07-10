from pymetamap import MetaMap
import os
import time


def main():

    metamap_base_dir = '/home/ivan/metamapdownload/public_mm/'
    metamap_bin_dir = 'bin/metamap20'

    metamap_pos_server_dir = 'bin/skrmedpostctl'
    metamap_wsd_server_dir = 'bin/wsdserverctl'

    os.system(metamap_base_dir + metamap_pos_server_dir + ' start') # Part of speech tagger
    os.system(metamap_base_dir + metamap_wsd_server_dir + ' start') # Word sense disambiguation 

    time.sleep(10)

    metam = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)

    note_list = ["headache", "atrial fibrilation", "heartache"]

    cons, errs = metam.extract_concepts(note_list)

    # Look at the output:
    print(cons)

    os.system(metamap_base_dir + metamap_pos_server_dir + ' stop') # Part of speech tagger
    os.system(metamap_base_dir + metamap_wsd_server_dir + ' stop') # Word sense disambiguation 

if __name__ == "__main__":
    main()