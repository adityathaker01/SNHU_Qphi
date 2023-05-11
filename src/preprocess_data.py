import pandas as pd
import gensim
import itertools
import csv
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
import config

def load_data(file_path):
    """
    This function loads the data from tsv file to a pandas Dataframe

    Parameters: 
    1. file_path(str) : path of the tsv file

    Returns: pandas Dataframe
    """
    try:
        data = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', quoting=csv.QUOTE_NONE)
        data.drop(['#1 ID','#2 ID'], axis = 1 ,inplace = True)

        return data

    except Exception as e:

        return "Exception: " + str(e)
    

def extract_sentences(train_data, test_data):
    """
    Extract sentences for making word2vec model.

    Parameters:
    1. train_data: training data loaded in dataframe
    2. test_data: test data loaded in dataframe
    """

    for dataset in [train_data, test_data]:
        for i, row in dataset.iterrows():
            if row['#1 String']:
                yield gensim.utils.simple_preprocess(row['#1 String'])
            if row['#2 String']:
                yield gensim.utils.simple_preprocess(row['#2 String'])
                    
def text_to_word(text):
    """
    Convert texts to a list of words
    """
    text = str(text)
    text = text.lower()
    text = text.split()

    return text
    
def make_w2v_embeddings(df, embedding_dim, EMB_MODEL_SAVE_PATH):
    """
    Converts text to embeddings
    
    Parameters:
    1. df: dataframe where training/test data is stored
    2. embedding_dim: dimention of the embedding to be generated
    """
    vocabs = {}
    vocabs_cnt = 0
    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Stopwords
    stops = set(stopwords.words('english'))

    # Load word2vec
    word2vec = gensim.models.word2vec.Word2Vec.load(EMB_MODEL_SAVE_PATH).wv

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 250 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both sentences of the row
        for sentence in ['#1 String', '#2 String']:

            s2n = []

            for word in text_to_word(row[sentence]):
                # Check for unwanted words
                if word in stops:
                    continue

                # If a word is missing from word2vec model.
                if word not in word2vec.key_to_index:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    s2n.append(vocabs_cnt)
                else:
                    s2n.append(vocabs[word])
        
            # Append sentence as number representation
            df.at[index, sentence] = s2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.key_to_index:
            embeddings[index] = word2vec.get_vector(word)
    del word2vec

    return df, embeddings
    
    
def split_and_zero_padding(df, max_seq_length):
    """
    Transforms list of sequences to 2D numpy array
    
    Parameters:
    1. df : dataframe where training/test data is stored
    2. max_seq_length : maximum length of all sequences
    """
    # Split to dicts
    X = {'left': df['#1 String'], 'right': df['#2 String']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = tf.keras.utils.pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset
    
def load_preprocess_data(TRAINING_DATA_PATH, TEST_DATA_PATH, EMB_MODEL_SAVE_PATH, embedding_dim, max_seq_length):
    """
    This function loads the data, preprocesses the data and splits the data into train and validation set
    """   
    #Load Data
    train_data = load_data(TRAINING_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)

    #Creating W2V model
    documents = list(extract_sentences(train_data, test_data))
    model = gensim.models.Word2Vec(documents, vector_size=50)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.save(EMB_MODEL_SAVE_PATH)

    print("Embedding Train Data")
    train_df, embeddings = make_w2v_embeddings(train_data, embedding_dim, EMB_MODEL_SAVE_PATH)

    print("Embedding Test Data")
    test_df, embeddings_test = make_w2v_embeddings(test_data, embedding_dim, EMB_MODEL_SAVE_PATH)

    #splitting into train and validation
    validation_size = int(len(train_df) * 0.1)

    X = train_df[['#1 String', '#2 String']]
    Y = train_df['Quality']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    #splitting and zero padding
    X_test = split_and_zero_padding(test_df, max_seq_length)
    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    Y_train = Y_train.values
    Y_validation = Y_validation.values
    Y_test = list(test_df['Quality'])

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test, embeddings
