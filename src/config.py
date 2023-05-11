# Data Paths
TRAINING_DATA_PATH = 'glue_data/MRPC/train.tsv'
TEST_DATA_PATH = 'glue_data/MRPC/dev.tsv'
# Model save Paths
MODEL_SAVE_PATH = './Models/LSTM.h5'
EMB_MODEL_SAVE_PATH = './Models/mrpc.w2v'
# W2V model params
embedding_dim = 50
max_seq_length = 20
#LSTM model params
batch_size = 100
n_epoch = 50
n_hidden = 64