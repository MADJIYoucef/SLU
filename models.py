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
from sklearn.metrics import log_loss, classification_report,accuracy_score,f1_score
from dataset import *

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class SlotFilingModel(object):
    
    def __init__(self, vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True):
        self.model = self.model(vocab_size,tag_vocab_size,max_length_sequence,embedding_size)
        self.model_path = model_path+'.h5'
        self.batch_size =batch_size
        self.nb_epochs = nb_epochs
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.early_stop = early_stop
        self.idx2tags = idx2tags
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy',f1,recall,precision])
        self.model.summary()
    
    def model(self,vocab_size,tag_vocab_size,max_length_sequence,embedding_size):
        pass
    
    def fit(self,X_train,Y_train,X_valid,Y_valid):
        early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=4,verbose=1)
        callbacks_list = [early_stopping]
        if self.early_stop :
            self.model.fit(X_train,Y_train,batch_size=self.batch_size,epochs=self.nb_epochs,callbacks=callbacks_list,validation_data=[X_valid,Y_valid],shuffle=True)
        else:
            self.model.fit(X_train,Y_train,batch_size=self.batch_size,epochs=self.nb_epochs,validation_data=[X_valid,Y_valid],shuffle=True)
        self.save(self.model_path)
        
    def evaluate(self,X,Y):
        if os.path.exists(self.model_path):
            self.load(self.model_path)
        loss, acc,f1,recall,precision = self.model.evaluate(X, Y)
        print("accuracy = {} - f1-score = {} - recall = {} - precision = {}".format(acc,f1,recall,precision))        
        
    def predict(self,X):
        if os.path.exists(self.model_path):
            self.load(self.model_path)
        res = []
        preds = self.model.predict(X) 
        for exemple in preds :
            res_temp = []
            for timestamp in exemple :
                res_temp.append(self.idx2tags[np.argmax(timestamp)])
            res.append(res_temp)    
        return res
    
    def predict_no_padding(self,X,Y):
        if os.path.exists(self.model_path):
            self.load(self.model_path)
        preds = self.model.predict(X)
        res_total = []
        pre_total = []
        res_intent = []
        pre_intent = []
        res_slot = []
        pre_slot = []
        cpt = 0
        p=0
        for i,y in enumerate(Y):
            for j,a in enumerate(y):
                t = np.argmax(a)
                if t != 0:
                    if np.argmax(Y[i][j]) == 2 :
                        p+=1
                    if X[i][j] == 12:
                        res_intent.append(t)
                        pre_intent.append(np.argmax(preds[i][j]))
                    else :
                        res_slot.append(t)
                        pre_slot.append(np.argmax(preds[i][j]))
                    res_total.append(t)
                    pre_total.append(np.argmax(preds[i][j]))
                else:
                    cpt+=1
        print("Pourcentage de la classe <pad> :",cpt/(len(X[0])*len(X)))
        print("Pourcentage de la classe O :",p/(len(Y[0])*len(Y)))
        print("Accuracy intent :",accuracy_score(res_intent,pre_intent))
        print("F1 score intent :",f1_score(res_intent,pre_intent,average='weighted'))
        print("Accuracy slot :",accuracy_score(res_slot,pre_slot))
        print("F1 score slot :",f1_score(res_slot,pre_slot,average='weighted'))
        print('--------------------------------------------------')
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

        
        
    def predict(self,X):
        if os.path.exists(self.model_path):
            self.load(self.model_path)
        res = []
        preds = self.model.predict(X) 
        for exemple in preds :
            res_temp = []
            for timestamp in exemple :
                res_temp.append(self.idx2tags[np.argmax(timestamp)])
            res.append(' '.join(res_temp))    
        return res
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# ## 2 Layer GRU :


class TwoGruModel(SlotFilingModel):
    
    def __init__(self, vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True):
        super().__init__(vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True)
        
    def model(self,vocab_size,tag_vocab_size,max_length_sequence,embedding_size):
        
        X_input = Input(shape=(max_length_sequence,), dtype='int32')
        X = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length_sequence, mask_zero = True)(X_input)
        
        X = GRU(units = 150, return_sequences = True)(X)    # GRU (use 150 units and return the sequences)
        X = Dropout(0.8)(X)                                 # dropout (use 0.8)
        X = BatchNormalization()(X)                         # Batch normalization

        X = GRU(units = 150, return_sequences = True)(X)    # GRU (use 150 units and return the sequences)
        X = Dropout(0.8)(X)                                 # dropout (use 0.8)
        X = BatchNormalization()(X)                         # Batch normalization
        X = Dropout(0.8)(X)                                 # dropout (use 0.8)

        X = TimeDistributed(Dense(tag_vocab_size, activation = "softmax"))(X) # time distributed  (sigmoid)
        model = Model(inputs = X_input, outputs = X)
        return model 

# ## Bidirectional LSTM

class BidirectionalLSTMModel(SlotFilingModel):
    
    def __init__(self, vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True):
        super().__init__(vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True)
        
    def model(self,vocab_size,tag_vocab_size,max_length_sequence,embedding_size):
        X_input = Input(shape=(max_length_sequence,), dtype='int32')
        X = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length_sequence, mask_zero = True)(X_input)
        A = LSTM(100, return_sequences=True, init='glorot_uniform', activation='relu')(X)
        B = LSTM(100, return_sequences=True, init='glorot_uniform', activation='relu', go_backwards=True)(X)
        res = concatenate([A, B])
        output = TimeDistributed(Dense(tag_vocab_size, activation = "softmax"))(res) # time distributed  (sigmoid)
        model = Model(inputs = X_input, outputs = output)
        return model


# ## Sequence to Sequence Model


class SeqToSeqModel(SlotFilingModel):
    
    def __init__(self, vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True):
        super().__init__(vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True)
        
    def model(self,vocab_size,tag_vocab_size,max_length_sequence,embedding_size):
        X_input = Input(shape=(max_length_sequence,), dtype='int32')
        X = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length_sequence, mask_zero = True)(X_input)
        forward = GRU(150,  return_sequences=False, init='glorot_uniform', activation='relu')(X)
        backward = GRU(150, return_sequences=False, init='glorot_uniform', activation='relu', go_backwards=True)(X)                                 # dropout (use 0.8)
        forward_target = GRU(150, return_sequences=True, init='glorot_uniform', activation='relu')(X)
        backward_target = GRU(150, return_sequences=True, init='glorot_uniform', activation='relu', go_backwards=True)(X)
        encoder = concatenate([forward, backward])
        target = concatenate([forward_target, backward_target])
        encoder = RepeatVector(max_length_sequence)(encoder)
        tagger = concatenate([encoder, target])
        output = TimeDistributed(Dense(tag_vocab_size, activation = "softmax"))(tagger) # time distributed  (sigmoid)
        model = Model(inputs = X_input, outputs = output)
        return model


# ## Conv1 Encoder Sequence to Sequence



class ConvEncoderSeq2SeqModel(SlotFilingModel):
    
    def __init__(self, vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True):
        super().__init__(vocab_size, tag_vocab_size, max_length_sequence, embedding_size,idx2tags,model_path, batch_size= 32, nb_epochs=10, early_stop=True)
        
    def model(self,vocab_size,tag_vocab_size,max_length_sequence,embedding_size):
        X_input = Input(shape=(max_length_sequence,), dtype='int32')
        X = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length_sequence)(X_input)
        X = Convolution1D(50, 3, border_mode='same', input_shape=(48,50))(X)
        forward = GRU(150,  return_sequences=False, init='glorot_uniform', activation='relu')(X)
        backward = GRU(150, return_sequences=False, init='glorot_uniform', activation='relu', go_backwards=True)(X)                                 # dropout (use 0.8)
        forward_target = GRU(150, return_sequences=True, init='glorot_uniform', activation='relu')(X)
        backward_target = GRU(150, return_sequences=True, init='glorot_uniform', activation='relu', go_backwards=True)(X)
        encoder = concatenate([forward, backward])
        target = concatenate([forward_target, backward_target])
        encoder = RepeatVector(max_length_sequence)(encoder)
        tagger = concatenate([encoder, target])
        output = TimeDistributed(Dense(tag_vocab_size, activation = "softmax"))(tagger) # time distributed  (sigmoid)
        model = Model(inputs = X_input, outputs = output)
        return model




