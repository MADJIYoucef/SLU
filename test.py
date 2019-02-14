
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
    corpus.test.pad_tags(time_length=max_len)
    X_train = corpus.train.padded_utterances
    X_valid = corpus.valid.padded_utterances
    X_test = corpus.test.padded_utterances
    Y_train = corpus.train.get_one_hot_tags(tag_vocab_size)
    Y_valid = corpus.valid.get_one_hot_tags(tag_vocab_size)
    Y_test = corpus.test.get_one_hot_tags(tag_vocab_size)
    idx_test = 500
    print(argparams)
    if argparams == '1' :
        print("Chargement du modèle 2 couche GRU ...")
        model_gru = TwoGruModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"two_gru_model")
        model_gru.predict_no_padding(X_test,Y_test) 
    elif argparams == '2': 
        print("Chargement du modèle BiLSTM ...")
        model_blstm = BidirectionalLSTMModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"blstm_model")
        model_blstm.predict_no_padding(X_test,Y_test) 
    elif argparams == '3':
        print("Chargement du modèle encodeur-decodeur ...")
        model_seq2seq = SeqToSeqModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"seq2seq_model")
        model_seq2seq.predict_no_padding(X_test,Y_test) 
    elif argparams == '4':
        print("Chargement du modèle conv-encodeur-decodeur ...")
        model_convSeq2Seq = ConvEncoderSeq2SeqModel(vocab_size,tag_vocab_size,max_len,embedding_size,corpus.dictionary.idx2tag,"convSeq2Seq_model")
        model_convSeq2Seq.predict_no_padding(X_test,Y_test) 
    else :
        print("SVP! il faut choisir un modèle entre 1 et 4")


