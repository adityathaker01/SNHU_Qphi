from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess_data import *
import json
import config

app = Flask(__name__)

#Loading configurations
EMB_MODEL_SAVE_PATH = config.EMB_MODEL_SAVE_PATH
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
embedding_dim = config.embedding_dim
max_seq_length = config.max_seq_length

#Load Model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

def predict(json_data):
    '''
    This Function is used to preprocess the request data and generate prediction
    
    '''
    
    #Reading Request data
    data_lstm = pd.DataFrame(json_data, index=[0])
    
    #Preprocessing Request data
    data_lstm, embeddings_test = make_w2v_embeddings(data_lstm, embedding_dim, EMB_MODEL_SAVE_PATH)
    data_lstm = split_and_zero_padding(data_lstm, max_seq_length)

    #Predict
    lstm_pred = model.predict([data_lstm['left'], data_lstm['right']])
    y_pred = np.where(lstm_pred > 0.5, 'similar', 'not similar')
    
    return y_pred

#Endpoint for prediction
@app.route('/prediction', methods=["POST"])
def make_prediction():
    
    #Read request data
    json_data = json.loads(request.data)
    
    #Generate prediction
    pred = predict(json_data)

    response = {'result' : str(pred)}
    
    return jsonify(response)

#Runner
if __name__ == "__main__":
    app.run(host='localhost',debug=True,port=8060)
