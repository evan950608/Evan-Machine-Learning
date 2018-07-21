# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:19:29 2018

@author: evan9
"""

import pickle
import urllib.request
import os
import tarfile
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
import inspect
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding

def save_pickle(fullname, value):
    with open(fullname, 'wb') as fileobj:
        pickle.dump(value, fileobj) #, protocol=pickle.HIGHEST_PROTOCOL

def load_pickle(fullname):
    with open(fullname, 'rb') as fileobj:
        value = pickle.load(fileobj)    
    return value

###Preprocess Data###
#Regular Expression
def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    #filetype is 'train' or 'test'
    path = r"C:\PyCodes\Keras\File\aclImdb"
    file_list = []
    all_labels = []
    all_texts = []
    
    positive_path = r'{0}\{1}\pos'.format(path, filetype)
    print('Positive path:', positive_path)
    for item in os.listdir(positive_path):
        file_list.append(r'{0}\{1}'.format(positive_path, item))
        all_labels.append(1)
    
    negative_path = r'{0}\{1}\neg'.format(path, filetype)
    print('Negative path:', negative_path)
    for item in os.listdir(negative_path):
        file_list.append(r'{0}\{1}'.format(negative_path, item))
        all_labels.append(0)
    
    #all_labels = ([1] * 12500 + [0] * 12500)
    print('Reading:', filetype, 'File quantity:', len(file_list))

    for file in file_list:
        with open(file, encoding='utf8') as file_input:
            all_texts.append(remove_tags(' '.join(file_input.readlines())))
    
    return all_texts, all_labels


def create_token(train_text):
    token = Tokenizer(num_words=2000)
    token.fit_on_texts(train_text)
    
    return token


def sequence_and_pad(token, text):
    sequenced = token.texts_to_sequences(text)
    padded = sequence.pad_sequences(sequenced, maxlen=100)
    
    return padded


def create_model():
    model = Sequential()
    
    #embedding layer
    model.add(Embedding(output_dim=32, 
                        input_dim=2000, 
                        ))
    
    return model



def main():
#    x_text_train, y_label_train = read_files('train')
#    x_text_test, y_label_test = read_files('test')
#    
#    data_plain = (x_text_train, y_label_train, x_text_test, y_label_test)
#    save_pickle(name_plain, data_plain)
    (x_text_train, y_label_train, x_text_test, y_label_test) = load_pickle(name_plain)
#    
#    token = create_token(x_text_train)
#    x_text_train_normalized = sequence_and_pad(token, x_text_train)
#    x_text_test_normalized = sequence_and_pad(token, x_text_test)
    
#    data_normalized = (x_text_train_normalized, x_text_test_normalized)
#    save_pickle(name_normalized, data_normalized)
    (x_text_train_normalized, x_text_test_normalized) = load_pickle(name_normalized)
    
    local_vars = inspect.currentframe().f_locals
    return local_vars


if __name__ == '__main__':
    local_vars = {}
    local_vars_detail = {}
    name_plain = r'c:\PyCodes\Keras\File\imdb_plain.pkl'
    name_normalized = r'c:\PyCodes\Keras\File\imdb_normalized.pkl'
    
    local_vars = main()
    
    for key,value in sorted(local_vars.items()):
        if type(value) == np.ndarray:
            local_vars_detail[key] = ['np.array', value.shape]
#        elif type(value) == pd.core.frame.DataFrame:
#            local_vars_detail[key] = ['pd.dataframe', value.shape]
        elif (key == 'model' or 
              key == 'train_history' or 
              key == 'token'):
            continue
        elif type(value) == Tokenizer or len(value) > 10:
            local_vars_detail[key] = [type(value), 'Unable to Display']
        else:
            local_vars_detail[key] = [type(value), value]

    print('Done')
    




