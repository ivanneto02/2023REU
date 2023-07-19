import pandas as pd

from gensim.models import Word2Vec

from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

def main():
    df = pd.read_csv("/mnt/c/Users/ivana/Desktop/Documents/Research/UCR/DS-PATH/working_dir/data/ready/ready_to_embed.csv", nrows=1000)
    
    sentences = df["definition"].to_list()
    sentences = [ remove_stopwords(s.lower()) for s in sentences ]

    model = Word2Vec(
        sentences = sentences,
        vector_size = 200,
        window = 7,
        min_count = 1,
        workers = 4
    )

    # train the models
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs = 10
    )

    # model.save("./src/scripting/saves/word2vec.model")

    print(model.wv)
    print(model.wv["treatment"])

if __name__ == "__main__":
    main()