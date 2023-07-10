import os
from dotenv import load_dotenv

load_dotenv()

# PREPROCESSING VARIABLES
SCRAPED_DATA_PATH = os.environ["_SCRAPED_DATA_PATH"]
SCRAPED_DATA_FILE = os.environ["_SCRAPED_DATA_FILE"]
UMLS_UNAME        = os.environ["_UMLS_UNAME"]
UMLS_PASSW        = os.environ["_UMLS_PASSW"]
SAVE_DATA_PATH    = os.environ["_SAVE_DATA_PATH"]

# EMBEDDING VARIABLES
READY_DATA_PATH   = os.environ["_READY_DATA_PATH"]
READY_DATA_FILE   = os.environ["_READY_DATA_FILE"]
EMBED_SAVE_FILE   = os.environ["_EMBED_SAVE_FILE"]

# ANALYSIS VARIABLES
# - nothing yet :(