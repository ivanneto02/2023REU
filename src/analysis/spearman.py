import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.test.utils import common_texts

from config import *
import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

# Used for spearman correlation, each list in tuple represents a term
# Credit to Gwen Kiler
term_list = [(["renal","failure"], ['kidney','failure']),
              (['heart'], ['myocardium']),
              (['stroke'], ['infarct']),
              (['abortion'], ['miscarriage']),
              (['delusion'], ['schizophrenia']),
              (['congestive', 'heart', 'failure'], ['pulmonary', 'edema']),
              (['metastasis'], ['adenocarcinoma']),
              (['calcification'], ['stenosis']),
              (['diarrhea'], ['stomach', 'cramps']),
              (['mitral', 'stenosis'], ['atrial', 'fibrillation']),
              (['chronic', 'obstructive', 'pulmonary', 'disease'], ['lung', 'infiltrates']),
              (['rheumatoid', 'arthritis'], ['lupus']),
              (['brain', 'tumor'], ['intracranial', 'hemorrhage']),
              (['carpel', 'tunnel', 'syndrome'], ['osteoarthritis']),
              (['diabetes', 'mellitus'], ['hypertension']),
              (['acne'], ['syringe']),
              (['antibiotic'], ['allergy']),
              (['cortisone'], ['total', 'knee', 'replacement']),
              (['pulmonary', 'embolus'], ['myocardial', 'infarction']),
              (['pulmonary', 'fibrosis'], ['lung', 'cancer']),
              (['cholangiocarcinoma'], ['colonoscopy']),
              (['lymphoid', 'hyperplasia'], ['laryngeal', 'cancer']),
              (['multiple', 'sclerosis'], ['psychosis']),
              (['appendicitis'], ['osteoporosis']),
              (['rectal', 'polyp'], ['aorta']),
              (['xerostomia'], ['alcoholic', 'cirrhosis']),
              (['peptic', 'ulcer', 'disease'], ['myopia']),
              (['depression'], ['cellulites']),
              (['varicose', 'vein'], ['entire', 'knee', 'meniscus']),
              (['hyperlipidemia'], ['metastasis'])]

# BIG credit to Gwen Kiler, she showed me this portion in her code
# Also saving me a lot of time
def test(model):
    similarity_list = []
    for cui1,cui2 in term_list:
        try:
            vec1 = model.infer_vector(cui1)
        except:
            print(f"Could not find {cui1}")
            continue
        try:
            vec2 = model.infer_vector(cui2)
        except:
            print(f"Could not find {cui2}")
            continue
        similarity_list.append(((cui1,cui2), cosine_similarity([vec1],[vec2])))

    similarity_list.sort(key=lambda x:x[1][0][0],reverse=True)
    cui_model_list = [x[0] for x in similarity_list]

    my_ranks = []
    for cui_pair in term_list:
        try:
            cui_model_list.index(cui_pair)+1
        except:
            term_list.remove(cui_pair)
            print(cui_pair)

    for cui_pair in term_list:
        try:
            my_ranks.append(cui_model_list.index(cui_pair)+1)
        except:
            term_list.remove(cui_pair)
        gold_ranks = [i for i in range(1,len(term_list)+1)]

    return spearmanr(my_ranks,gold_ranks)[0]

# Credit to Gwen Kiler for showing me this part and how she did it
def spearman():
    print(" > Spearman's Rank Correlation...")

    print("     - Loading model")
    model = Doc2Vec.load(MODEL_SAVE_PATH + MODEL_DOC2VEC)

    print("     - Starting Spearman Correlation Test")
    sum = 0
    for i in tqdm.tqdm(range(0,1000)):
        sum += test(model)
    print(sum/1000)