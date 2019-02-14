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

class Dictionary(object):
    # to store the vocabulary
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']
        self.tag2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2tag = ['<pad>', '<unk>'] 
        self.word_vocab_size = 2 
        self.tag_vocab_size  = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_vocab_size+=1
        return self.word2idx[word]
    
    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1
            self.tag_vocab_size+=1
        return self.tag2idx[tag]   

    def __len__(self):
        return len(self.idx2word)

class GolveEmbbedings(object): 
    
    def __init__(self,file_name):
        self.words,self.embeddings = self.loadEmbeddings(file_name)
        
    def loadEmbeddings(self, path):
        with open(path, 'r') as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        return words, word_to_vec_map

class SlotDataset(object):
    
    def __init__(self):
        self.utterances = []
        self.tags =[]
        self.padded_utterances = []
        self.padded_tags =[]
        self.one_hot_tags =[]
        
    def add_utterance(self, utt):
        self.utterances.append(utt)
        
    def add_slots(self, slots):
        self.tags.append(slots)
        
    def pad_utts(self,time_length=100):
        self.padded_utterances = sequence.pad_sequences(self.utterances, maxlen=time_length, dtype='int32', padding='pre')
        return self.padded_utterances
    
    def pad_tags(self,time_length=100):
        self.padded_tags = sequence.pad_sequences(self.tags, maxlen=time_length, dtype='int32', padding='pre')
        return self.padded_tags
    
    def get_one_hot_tags(self,tag_vocab_size=100):
        self.one_hot_tags =[]
        for i,sentence  in  enumerate(self.padded_tags):
            temp_sent = self.convert_to_one_hot(sentence,tag_vocab_size)
            self.one_hot_tags.append(temp_sent)
        self.one_hot_tags = np.array(self.one_hot_tags)
        return self.one_hot_tags
    
    def convert_to_one_hot(self,Y, C):
        Y = np.eye(C)[Y.reshape(-1)]
        return Y
    
    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    
    def __init__(self, data_dir='data',embeddings_file="glove.6B.50d.txt", train_file="atis-2.train.w-intent.iob", valid_file="atis-2.dev.w-intent.iob" ,test_file="atis.test.w-intent.iob",):
        self.train = SlotDataset()
        self.valid = SlotDataset()
        self.test = SlotDataset()
        
        self.dictionary = Dictionary()
        #self.embeddings = GolveEmbbedings(os.path.join(data_dir, embeddings_file))
        self.load_train(os.path.join(data_dir, train_file))
        self.load_valid(os.path.join(data_dir, valid_file))
        self.load_test(os.path.join(data_dir, test_file))
        
    def load_train(self, path):
        for line in open(path, 'r'):
            words=line.split('\t')[0].strip().split()
            tags=line.split('\t')[1].strip().split()
            temp_utt = list()
            temp_tags = list()
            for i in range(len(words)):
                id_word = self.dictionary.add_word(words[i])
                id_tag  = self.dictionary.add_tag(tags[i])
                temp_utt.append(id_word)
                temp_tags.append(id_tag)
            self.train.add_utterance(np.array(temp_utt))
            self.train.add_slots(np.array(temp_tags))
    
    def load_valid(self, path):
        for line in open(path, 'r'):
            words=line.split('\t')[0].strip().split()
            tags =line.split('\t')[1].strip().split()
            temp_utt = list()
            temp_tags = list()
            for i in range(len(words)):
                if words[i] not in self.dictionary.word2idx :
                    temp_utt.append(1) 
                else:
                    temp_utt.append(self.dictionary.word2idx[words[i]])
                if tags[i] not in self.dictionary.tag2idx :
                    temp_tags.append(1) 
                else:
                    temp_tags.append(self.dictionary.tag2idx[tags[i]])
            self.valid.add_utterance(np.array(temp_utt))
            self.valid.add_slots(np.array(temp_tags))
            
    def load_test(self, path):
        for line in open(path, 'r'):
            words=line.split('\t')[0].strip().split()
            tags =line.split('\t')[1].strip().split()
            temp_utt = list()
            temp_tags = list()
            for i in range(len(words)):
                if words[i] not in self.dictionary.word2idx :
                    temp_utt.append(1) 
                else:
                    temp_utt.append(self.dictionary.word2idx[words[i]])
                if tags[i] not in self.dictionary.tag2idx :
                    temp_tags.append(1) 
                else:
                    temp_tags.append(self.dictionary.tag2idx[tags[i]])
            self.test.add_utterance(np.array(temp_utt))
            self.test.add_slots(np.array(temp_tags))


