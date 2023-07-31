import src.analysis as analysis

def main():
    '''
        Initializes entire analysis process. This will:
            (1) Use spearman correlation to evaluate the Doc2Vec model
    '''

    try:
        # ---------
        print("> Starting [TRAINING] process")
        # Training steps

        print("     - Starting spearman correlation process")
        analysis.spearman()
        print("> Done!")

    except Exception as e:
        print("   ! Something went wrong !")
        print(e)

if __name__ == "__main__":
    main()