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

# WIKIPEDIA DATA VARIABLES
WIKI_DATA_PATH    = os.environ["_WIKI_DATA_PATH"]
WIKI_DATA_FILE    = os.environ["_WIKI_DATA_FILE"]
PARALLEL_SIZE     = int(os.environ["_PARALLEL_SIZE"])

# ANALYSIS VARIABLES
# - nothing yet :(

# TESTING PURPOSES
NROWS = 100
if os.environ["_NROWS"] == "None":
    NROWS = None
else:
    NROWS = int(os.environ["_NROWS"])

CHUNKSIZE        = int(os.environ["_CHUNKSIZE"])

# RELEVANT PARAMETERS
KEEPWORDS        = int(os.environ["_KEEPWORDS"])

# DOC2VEC Hyperparameters
LAYER_SIZE       = int(os.environ["_LAYER_SIZE"])
WINDOW_SIZE      = int(os.environ["_WINDOW_SIZE"])
LEARNING_RATE    = float(os.environ["_LEARNING_RATE"])
EPOCHS           = int(os.environ["_EPOCHS"])
WORKERS          = int(os.environ["_WORKERS"])
MIN_COUNT        = int(os.environ["_MIN_COUNT"])
NEGATIVE_SAMPLES = int(os.environ["_NEGATIVE_SAMPLES"])
NS_EXPONENT      = float(os.environ["_NS_EXPONENT"])
DM               = int(os.environ["_DM"])
HS               = int(os.environ["_HS"])
DM_CONCAT        = int(os.environ["_DM_CONCAT"])
DM_MEAN          = int(os.environ["_DM_MEAN"])