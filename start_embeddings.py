import src.embeddings as embeddings

def main():
    '''
        Initializes entire embedding process. This will:
            (1) Use word2vec (CBOW + Skip-Gram) to pre-process every sentence
            (2) Use doc2vec (PV-DBOW) to create concept embeddings
    '''

    try:
        # ---------
        print("> Starting [TRAINING] process")
        # Training steps
        embeddings.word2vec_pre()                   # Step in order to retain co-ocurrance information
        embeddings.doc2vec_train()                  # Step in order to actually classify CUIs into embeds

        print("> Done!")

    except Exception as e:
        print("   ! Something went wrong !")
        print(e)

if __name__ == "__main__":
    main()