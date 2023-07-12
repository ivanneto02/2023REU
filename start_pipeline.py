import src.preprocessing as preprocessing
import src.embeddings as embeddings
import src.analysis as analysis

def main():
    '''
        Initializes entire pipeline process. This will:
            (1) Combine all data sources (including UMLS) into a singular big dataframe.
            (2) Extract all of the definitions from the scraped sources
            (3) Tag all of the non-UMLS sources with CUIs with metamap
            (4) Drop all of the non-tagged rows
            (5) Limit every definition to 5 sentences or less
            (6) Use word2vec (CBOW + Skip-Gram) to pre-process every sentence
            (7) Use doc2vec (PV-DBOW) to create concept embeddings
            (8) Calculate the Spearman's Rank Coefficient 
    '''

    try:

        print(f"> Starting pipeline process")

        print("> Starting [PREPROCESSING] process")
        # Preprocessing steps
        preprocessing.combine_data()

        ## Important step - Later will include another version of this step,                    PLEASE READ     **
        ## which uses source-based html removal. Currently I just remove ALL
        ## html tags and leave only the raw text. This is for testing purposes
        ## for the pipeline
        preprocessing.extract_nonumls_definitions() # May take a while
        ###

        ## Important step - Later will include a non-metamap version, with                      PLEASE READ     **
        ## just an edit distance based tagging of CUIs. Terms that look like
        ## each other may also be related to one another.
        ### ** For now only uses MetaMap to tag exact terms and ngrams.
        preprocessing.tag_terms()                   # May take a while

        '''
            NOTE: `extract_nonumls_definitions` is strategically placed before `tag_terms`      PLEASE READ     **
            because `tag_terms` explodes the table into many times bigger. If done in the
            opposite way, it would take MUCH longer to execute for the same result! BEWARE.
        '''

        # Will drop NA rows associated with CUIs and definitions
        preprocessing.drop_nas()

        ## Important step - Later will decide whether including the entire                      PLEASE READ     **
        ## definition somehow affects the model
        preprocessing.limit_definitions()

        ## Important step - Later will decide how and why we normalize the
        ## data. Currently we are thinking of using AdaGrad as normalization
        ## step between different sources but will have to see.
        preprocessing.normalize() # This is probably very important
        
        # ---------
        print("> Starting [TRAINING] process")
        # Training steps
        embeddings.word2vec_pre()                   # Step in order to retain co-ocurrance information
        embeddings.doc2vec_train()                  # Step in order to actually classify CUIs into embeds

        # ---------
        print("> Starting [ANALYSIS] process")
        # Analysis steps
        analysis.spearman()                         # Will determine how similar embeddings are to close CUIs

        print("> Done!")

    except Exception as e:
        print("   ! Something went wrong !")
        print(e)

if __name__ == "__main__":
    main()