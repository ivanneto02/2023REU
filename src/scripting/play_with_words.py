import word2vec
import pandas as pd

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from keras import layers

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# HYPERPARAMETERS
WINDOW_SIZE          = 7
SEED                 = 0 # For reproducibility
BATCH_SIZE           = 1024
BUFFER_SIZE          = 10000
AUTOTUNE             = tf.data.AUTOTUNE
EMBEDDING_DIMENSION  = 128
EPOCHS               = 10
NUM_NS               = 4
VOCAB_SIZE           = 4096
VECTORIZATION_LENGTH = 50

### CREDIT TO AUTHORS OF https://www.tensorflow.org/tutorials/text/word2vec#model_and_training FOR HELPING ME
### TO WRITE THIS CODE.

def generate_training_samples(sequences, vocab_size, num_ns=3, window_size=5, seed=0):
    # Stores the final data that we will use :)
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    # Helps optimize I think
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    print("Creating training data...")
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
            seed=seed
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            
            # Figure out the context class
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1
            )
            
            # Figure out the negative samples
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling"
            )

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def main():
    df = pd.read_csv("/mnt/c/Users/ivana/Desktop/Documents/Research/UCR/DS-PATH/working_dir/data/ready/ready_to_embed.csv", nrows=None)

    stop_words = set(stopwords.words('english'))
    print(stop_words)

    # Grab all sentences
    sentences = df["definition"].to_numpy()

    # Now, create a custom standardization function to lowercase the text and
    # remove punctuation.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        no_stopwords = tf.strings.regex_replace(lowercase, r'\b(' + r'|'.join(stop_words) + r')\b\s*', "")
        strings = tf.strings.regex_replace(no_stopwords, r"\.|\,|\;|\!|\?|\:", "")
        strings = tf.strings.regex_replace(strings, r'[%s]' % re.escape(string.punctuation), '')
        strings = tf.strings.regex_replace(strings, r"\ +", " ")
        strings = tf.strings.strip(strings)
        return strings

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=VECTORIZATION_LENGTH
    )
    
    # Create vocabulary
    sentences = tf.data.Dataset.from_tensor_slices(sentences)
    vectorize_layer.adapt(sentences.batch(BATCH_SIZE))

    # Save the created vocabulary for reference.
    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:20])

    # Vectorize the data in text_ds.
    text_vector_ds = sentences.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences))

    # print("Cleaning up sentences...")
    # for sentence in tqdm.tqdm(sentences):

    #     # Strip unwanted sides and remove characters
    #     sentence = sentence.strip().lower()              
    #     sentence = re.sub("\.|\,|\;|\!|\?|\:", "", sentence)

    #     # Split by several standards (symbols)
    #     tokens = list(re.split("\-|\ |\[|\]|\(|\)|\<|\>", sentence)) # Split
    #     tokens = list(filter(lambda x: x != "", tokens))             # Remove ''

    #     # Add token to vocab
    #     for token in tokens:
    #         if token not in vocab:
    #             vocab[token] = index
    #             index += 1
        
    #     # Add current sequence to sequences
    #     sequences.append([ vocab[word] for word in tokens ])

    # # Calculate for the generation of samples
    # vocabulary_size = len(vocab)

    # # Make inverse vocabulary for me to see
    # inverse_vocab = { index: token for token, index in vocab.items() }

    targets, contexts, labels = generate_training_samples(
        sequences   = sequences,
        vocab_size  = VOCAB_SIZE,
        num_ns      = NUM_NS,
        window_size = WINDOW_SIZE,
        seed        = SEED
    )

    # Print out some targets, contexts, labels
    # for target, context, label in zip(targets, contexts, labels):
    #     print(f"target_index    : {target}")
    #     print(f"target_word     : {inverse_vocab[target]}")
    #     print(f"context_indices : {context}")
    #     print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
    #     print(f"label           : {label}")
    
    # Print out the shapes to make sure I am right
    print(f"Targets: {np.shape(targets)}")
    print(f"Contexts: {np.shape(contexts)}")
    print(f"Labels: {np.shape(labels)}")

    # print(targets)
    # print(contexts)
    # print(labels)

    # Train model and save it
    model = train_word2vec(targets, contexts, labels, vocab_size=VOCAB_SIZE)

    # Get the weights and vocabulary
    weights = model.get_layer("w2v_embedding").get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    # Save the model
    out_v = io.open('./src/scripting/saves/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('./src/scripting/saves/metadata.tsv', 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=300, num_ns=5):
        super(Word2Vec, self).__init__()

        self.target_embedding = layers.Embedding(vocab_size,
                                    embedding_dim,
                                    input_length=1,
                                    name="w2v_embedding")

        self.context_embedding = layers.Embedding(vocab_size,
                                    embedding_dim,
                                    input_length=num_ns+1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+

        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        
        # target: (batch,)
        word_emb = self.target_embedding(target)

        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        
        # dots: (batch, context)
        return dots

def train_word2vec(targets, contexts, labels, vocab_size):
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    word2vec = Word2Vec(vocab_size, embedding_dim=EMBEDDING_DIMENSION, num_ns=NUM_NS)
    word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./src/scripting/training_logs")

    word2vec.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])

    # Return the model to access outside
    return word2vec
if __name__ == "__main__":
    main()