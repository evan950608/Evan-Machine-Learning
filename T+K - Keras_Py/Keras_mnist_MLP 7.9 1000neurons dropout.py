# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:00:23 2018

@author: evan9
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import matplotlib.pyplot as plt

def preprocess_data():
    np.random.seed(10)
    (x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
    
    x_train = x_train_image.reshape(60000, 784).astype('float32')
    x_test = x_test_image.reshape(10000, 784).astype('float32')
    x_train_normalize = x_train / 255
    x_test_normalize = x_test / 255
    y_train_onehot = np_utils.to_categorical(y_train_label)
    y_test_onehot = np_utils.to_categorical(y_test_label)
    
    return x_train_normalize, x_test_normalize, y_train_onehot, y_test_onehot


def create_model():
    model = Sequential()
    
    model.add(Dense(units=1000, 
                    input_dim=784, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, 
                    kernel_initializer='normal', 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, train_que, train_ans):
    train_history = model.fit(x=train_que, 
                              y=train_ans, 
                              validation_split=0.2, 
                              epochs=10, batch_size=200, verbose=2)
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


def evaluate_accuracy(model, test_que, test_ans):
    scores = model.evaluate(test_que, test_ans)
    print()
    print('accuracy= %.4f' % (scores[1],))
    
    return


def make_prediction(model, analyte):
    prediction = model.predict_classes(analyte)
    predicted_probability = model.predict(analyte)
    
    return prediction, predicted_probability


def save_model(model, model_name):
    model.save(model_name)

    del model
    

def import_model(model_name):
    model = load_model(model_name)
    print(model.summary())        

    return model


name = r"C:\PyCodes\Keras\File\H_Keras_mnist_MLP 7.9 1000neurons dropout.h5"
def main():
    already_trained = True
    
    (x_train_normalize, x_test_normalize, y_train_onehot, y_test_onehot) = preprocess_data()

    if already_trained == False:
        model = create_model()
        train_history = train_model(model, x_train_normalize, y_train_onehot)
        save_model(model, name)
        
    elif already_trained == True:
        model = import_model(name)

    evaluate_accuracy(model, x_test_normalize, y_test_onehot)
    
    return


if __name__ == '__main__':
    main()







