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
        preprocessing.tag_definitions()
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