import os
from dotenv import load_dotenv

load_dotenv()

# PREPROCESSING VARIABLES
SCRAPED_DATA_PATH = os.environ["_SCRAPED_DATA_PATH"]
SCRAPED_DATA_FILE = os.environ["_SCRAPED_DATA_FILE"]
UMLS_HOST         = os.environ["_UMLS_HOST"]
# UMLS_PORT         = os.environ["_UMLS_PORT"]
UMLS_DB           = os.environ["_UMLS_DB"]
UMLS_USER         = os.environ["_UMLS_USER"]
UMLS_PASSW        = os.environ["_UMLS_PASSW"]
SAVE_DATA_PATH    = os.environ["_SAVE_DATA_PATH"]
SAVE_DATA_FILE    = os.environ["_SAVE_DATA_FILE"]

# EMBEDDING VARIABLES
READY_DATA_PATH   = os.environ["_READY_DATA_PATH"]
READY_DATA_FILE   = os.environ["_READY_DATA_FILE"]
EMBED_DEF_FILE    = os.environ["_EMBED_DEF_FILE"]
EMBED_SAVE_FILE   = os.environ["_EMBED_SAVE_FILE"]

MODEL_SAVE_PATH   = os.environ["_MODEL_SAVE_PATH"]
MODEL_WORD2VEC    = os.environ["_MODEL_WORD2VEC"]
MODEL_DOC2VEC     = os.environ["_MODEL_DOC2VEC"]

# ANALYSIS VARIABLES
# - nothing yet :(

# TESTING PURPOSES
NROWS = 100
if os.environ["_NROWS"] == "None":
    NROWS = None
else:
    NROWS = int(os.environ["_NROWS"])

# RELEVANT PARAMETERS
KEEPWORDS         = int(os.environ["_KEEPWORDS"])