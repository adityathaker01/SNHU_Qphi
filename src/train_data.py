import pandas as pd
import gensim
import itertools
import csv
import numpy as np
import tensorflow as tf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from preprocess_data import *
import config
    

def train_model(TRAINING_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH, EMB_MODEL_SAVE_PATH, embedding_dim, max_seq_length, batch_size, n_epoch, n_hidden):
    
    '''
    Trains the LSTM model and evaluated the trained model on the test data
    
    Parameters:
    1. TRAINING_DATA_PATH: path of training data
    2. TEST_DATA_PATH: path of test data
    3. embedding_dim: Dimentions of the W2V embeddings to be generated
    4. max_seq_length: maximum requesnce length while padding
    5. batch_size: batch size to be used for minibatch gradient descent
    5. n_epochs: epoch while training the model
    6. n_hidden: no of hidden nodes in the model
    
    '''
    
    #Loading and preprocessing data
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test, embeddings = load_preprocess_data(TRAINING_DATA_PATH, TEST_DATA_PATH, EMB_MODEL_SAVE_PATH, embedding_dim, max_seq_length)
    
    #Creating Model
    model_lstm = Sequential()
    model_lstm.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
    model_lstm.add(LSTM(n_hidden, dropout=0.3))
    shared_model = model_lstm
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')
    #Merging the two LSTM networks
    merged_model = concatenate([shared_model(left_input), shared_model(right_input)])
    merged_model = Dense(n_hidden, activation='relu')(merged_model)
    merged_model = Dropout(0.3)(merged_model)
    preds = Dense(1, activation='sigmoid')(merged_model)
    model = Model(inputs=[left_input, right_input], outputs=preds)
    
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    
    #Early stopping
    early_stopping =EarlyStopping(monitor='val_loss', patience = 5)
    
    try:
        
        #Train Model
        hist = model.fit([X_train['left'], X_train['right']], Y_train,
                            batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                            callbacks=[early_stopping])
        
        #Evalution on Test Data
        prediction = model.predict([X_test['left'], X_test['right']])
        y_pred = np.where(prediction > 0.5, 1, 0)
        
        print("Recall:", recall_score(Y_test, y_pred))
        print("Precision:", precision_score(Y_test, y_pred))
        print("F1:", f1_score(Y_test, y_pred))
        print("Accuracy:", accuracy_score(Y_test, y_pred))
        
        print("Model Training Completed!!!")
        
        #Saving Model
        model.save(MODEL_SAVE_PATH)
        
        return "Model Training Completed!!!"
    
    except Exception as e:
        
        return "Model Training Failed due to the following exception : " + str(e)

if __name__=='__main__':
    
    TRAINING_DATA_PATH = config.TRAINING_DATA_PATH
    TEST_DATA_PATH = config.TEST_DATA_PATH
    EMB_MODEL_SAVE_PATH = config.EMB_MODEL_SAVE_PATH
    MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
    embedding_dim = config.embedding_dim
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    n_epoch = config.n_epoch
    n_hidden = config.n_hidden
    
    model_status = train_model(TRAINING_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH, EMB_MODEL_SAVE_PATH, embedding_dim, max_seq_length, batch_size, n_epoch, n_hidden)
    
