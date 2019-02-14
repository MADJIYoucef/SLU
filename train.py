
import os, sys, json
import numpy as np 
from scipy import io
from keras.preprocessing import sequence
from keras.layers import Input,concatenate, merge, Dense, Dropout, Activation, RepeatVector, Permute, Reshape, RepeatVector, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import  GRU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.callbacks import EarlyStopping
from dataset import *
from models import *
from sklearn.metrics import log_loss, classification_report,accuracy_score,f1_score
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', dest='models', type=str, default='3', help='Model a choisir')
    args = parser.parse_args()
    argparams = vars(args)["models"]
    corpus = Corpus()
    max_len  = max([ len(x) for x in  corpus.train.utterances])
    vocab_size = len(corpus.dictionary.word2idx)
    tag_vocab_size = len(corpus.dictionary.tag2idx)
    embedding_size = 50
    corpus.train.pad_utts(time_length=max_len)
    corpus.train.pad_tags(time_length=max_len)
    corpus.valid.pad_utts(time_length=max_len)
    corpus.valid.pad_tags(time_length=max_len)
    corpus.test.pad_utts(time_length=max_len)
    corpus.test.pad_tags(time_length=max_len)[0]
    X_train = corpus.train.padded_utterances
    X_valid = corpus.valid.padded_utterances
    X_test = corpus.test.padded_utterances
    print('Data set analysis')
    print("Train shape " + str(X_train.shape))
    print("Dev shape " + str(X_valid.shape))
    print("Test shape " + str(X_test.shape))
    Y_train = corpus.train.get_one_hot_tags(tag_vocab_size)
    Y_valid = corpus.valid.get_one_hot_tags(tag_vocab_size)
    Y_test = corpus.test.get_one_hot_tags(tag_vocab_size)
    print("Train label shape " + str(Y_train.shape))
    print("Dev label shape " + str(Y_valid.shape))
    print("Test label shape " + str(Y_test.shape))
    print('-----------------------------------------------------------------')

    if argparams == '1' :
        model_gru = TwoGruModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"two_gru_model")
        model_gru.fit(X_train,Y_train,X_valid,Y_valid)
    elif argparams == '2' :
       model_blstm = BidirectionalLSTMModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"blstm_model")
       model_blstm.fit(X_train,Y_train,X_valid,Y_valid)       
    elif argparams == '3' :
       model_seq2seq = SeqToSeqModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"seq2seq_model")
       model_seq2seq.fit(X_train,Y_train,X_valid,Y_valid)
    elif argparams == '4' :
       model_convSeq2Seq = ConvEncoderSeq2SeqModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"convSeq2Seq_model")
       model_convSeq2Seq.fit(X_train,Y_train,X_valid,Y_valid)
    else:
       print("SVP! il faut choisir un modèle entre 1 et 4")














