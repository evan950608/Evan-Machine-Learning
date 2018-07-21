# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:21:19 2018

@author: evan9
"""

import inspect
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
np.random.seed(10)

### SET UP ###
def preprocess_data(raw_df):
    # name
    df = raw_df.drop(['name'], axis=1)
    # age
    age_avg = df['age'].mean()
    df['age'] = df['age'].fillna(age_avg)
    # fare
    fare_avg = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_avg)
    # sex
    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
    # embarked
    x_df_onehot = pd.get_dummies(data=df, columns=['embarked'])
    
    # convert to array
    ndarray = x_df_onehot.values
    label = ndarray[:, 0]
    features = ndarray[:, 1:]
    
    # normalize features
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled_features = minmax_scale.fit_transform(features)
    
    return scaled_features, label


def create_dataframe(xls_filepath, ):
    all_df = pd.read_excel(xls_filepath)
    cols = ['survived', 
            'name', 
            'pclass' , 
            'sex', 
            'age', 
            'sibsp',
            'parch', 
            'fare', 
            'embarked']
    all_df = all_df[cols]
    
    #80% train data; 20% test data
    msk = np.random.rand(len(all_df)) < 0.8
    train_df = all_df[msk]
    test_df = all_df[~msk]
    print('total: {0}, train: {1}, test: {2}'.format(len(all_df), len(train_df), len(test_df)))
    
    return all_df, train_df, test_df


def create_model():
    model = Sequential()
    
    # input layer & hidden layer 1
    model.add(Dense(units=40, input_dim=9, 
                    kernel_initializer='uniform', 
                    activation='relu'))
    # hidden layer 2
    model.add(Dense(units=30, 
                    kernel_initializer='uniform', 
                    activation='relu'))
    # output layer
    model.add(Dense(units=1, 
                    kernel_initializer='uniform', 
                    activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def jack_rose_data():
    Jack = pd.Series([0 , 'Jack', 3, 'male'  , 23, 1, 0,  5.0000,'S'])
    Rose = pd.Series([1 , 'Rose', 1, 'female', 20, 1, 0, 100.0000,'S'])
    JR_df = pd.DataFrame([list(Jack), list(Rose)], 
                          columns=['survived', 'name','pclass', 'sex', 'age', 
                                   'sibsp','parch', 'fare','embarked'])
    return JR_df
    
### TRAINING ###
def train_model(model, train_que, train_ans):
    train_history = model.fit(x=train_que, 
                              y=train_ans, 
                              validation_split=0.1, 
                              epochs=30, batch_size=30, verbose=2)
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')    

    return train_history


def show_train_history(train_history, train_accuracy, validation_accuracy):
    plt.plot(train_history.history[train_accuracy])
    plt.plot(train_history.history[validation_accuracy])
    plt.title('Train History')
    plt.ylabel(train_accuracy)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    return

### SAVE & LOAD ###
def save_model(model, model_name):
    model.save(model_name)
    print('Saved to {0}'.format(model_name))

    del model
    return
    

def import_model(model_name):
    model = load_model(model_name)
    print(model.summary())
    print('Loaded from {0}'.format(model_name))

    return model

### PRESENTATION ###
def evaluate_accuracy(model, test_que, test_ans):
    scores = model.evaluate(test_que, test_ans)
    print('accuracy= %.4f' % (scores[1],))
    
    return


def make_prediction(model, test_que):
    prediction = model.predict_classes(test_que)
    predicted_probability = model.predict(test_que)
    
    return prediction, predicted_probability


### ========== ###

def main():
    global local_vars
    already_trained = False
    
    (all_df, train_df, test_df) = create_dataframe(filepath)
    (x_features_train, y_label_train) = preprocess_data(train_df)
    (x_features_test, y_label_test) = preprocess_data(test_df)
    
    if already_trained == False:
        model = create_model()
        train_history = train_model(model, x_features_train, y_label_train)
#        save_model(model, hname)
#    elif already_trained == True:
#        model = import_model(hname)
        
    evaluate_accuracy(model, x_features_test, y_label_test)
    
    JR_df = jack_rose_data()
    all_df = pd.concat([all_df, JR_df])
    (x_features_all, y_label_all) = preprocess_data(all_df)
    
    (all_prediction, all_predicted_probability) = make_prediction(model, x_features_all)
    probability_df = all_df
    probability_df.insert(len(all_df.columns), 'probability', all_predicted_probability)
    print(probability_df[:10])
    
    unlikely_death_df = probability_df[(probability_df['survived'] == 0) & 
                                       (probability_df['probability'] >= 0.9)]
    print(unlikely_death_df)
    
    local_vars = inspect.currentframe().f_locals
    return local_vars


if __name__ == '__main__':
    filepath = r"C:\PyCodes\Keras\File\titanic3.xls"
    hname = r"C:\PyCodes\Keras\File\_H_keras0.h5"
    local_vars = {}
    local_vars_detail = {}
    
    local_vars = main()
    
    for key,value in sorted(local_vars.items()):
        if type(value) == np.ndarray:
            local_vars_detail[key] = ['np.array', value.shape]
        elif type(value) == pd.core.frame.DataFrame:
            local_vars_detail[key] = ['pd.dataframe', value.shape]
        elif key == 'model' or key == 'train_history': continue
        else:
            local_vars_detail[key] = [type(value), value]
    
    for key,value in local_vars_detail.items():
        print(key,value)
    
#    input('Press enter to continue...')

    
    
    