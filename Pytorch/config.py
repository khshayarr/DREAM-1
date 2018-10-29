# -*- coding:utf-8 -*-
__author__ = 'Randolph'


class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = 'data/Train.json'
        self.VALIDATIONSET_DIR = 'data/Validation.json'
        self.TESTSET_DIR = 'data/Test.json'
        self.NEG_SAMPLES = 'data/neg_sample.pickle'
        self.MODEL_DIR = 'runs/'
        self.clip = 10
        self.epochs = 100
        self.batch_size = 256
        self.seq_len = 12
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'max'  # ['avg', 'max']
        self.rnn_type = 'GRU'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        self.dropout = 0.5
        self.num_product = 26991+1  # 商品数目，用于定义 Embedding Layer
        self.embedding_dim = 32  # 商品表征维数， 用于定义 Embedding Layer
