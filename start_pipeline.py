import src.preprocessing as preprocessing
import src.embeddings as embeddings
import src.analysis as analysis

def main():
    '''
        Initializes entire pipeline process. This will:
            (1) Combine all data sources (including UMLS) into a singular big dataframe.
            (2) Tag all of the non-UMLS sources with CUIs with metamap
            (3) Drop all of the non-tagged rows
            (4) Limit every definition to 5 sentences or less
            (5) Use word2vec (CBOW + Skip-Gram) to pre-process every sentence
            (6) Use doc2vec (PV-DBOW) to create concept embeddings
            (7) Calculate the Spearman's Rank Coefficient 
    '''

    try:

        print(f"> Starting pipeline process")

        print("> Starting [PREPROCESSING] process")
        # Preprocessing steps
        preprocessing.combine_data()

        ## Important step - Later will include a non-metamap version, with
        ## just an edit distance based tagging of CUIs. Terms that look like
        ## each other may also be related to one another.
        ### ** For now only uses MetaMap to tag exact terms and ngrams.
        preprocessing.tag_terms()

        ## Important step - Later will include another version of this step,
        ## which uses source-based html removal. Currently I just remove ALL
        ## html tags and leave only the raw text. This is for testing purposes
        ## for the pipeline
        preprocessing.extract_nonumls_definitions() # May take a while
        ###

        preprocessing.drop_nas()
        preprocessing.limit_definitions()
        preprocessing.normalize() # This is probably very important
        
        # ---------
        print("> Starting [TRAINING] process")
        # Training steps
        embeddings.word2vec_pre()
        embeddings.doc2vec_train()

        # ---------
        print("> Starting [ANALYSIS] process")
        # Analysis steps
        analysis.spearman()

        print("> Done!")

    except Exception as e:
        print("   ! Something went wrong !")
        print(e)

if __name__ == "__main__":
    main()