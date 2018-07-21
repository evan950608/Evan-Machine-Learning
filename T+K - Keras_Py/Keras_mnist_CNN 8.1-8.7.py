# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:47:16 2018

@author: evan9
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_data():
    np.random.seed(10)
    (x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
    
    x_train = x_train_image.reshape(x_train_image.shape[0], 28, 28, 1).astype('float32')
    # x_train.shape == (60000, 28, 28)
    x_test = x_test_image.reshape(x_test_image.shape[0], 28, 28, 1).astype('float32')
    # x_test.shape == (10000, 28, 28)
    
    x_train_normalize = x_train / 255
    x_test_normalize = x_test / 255
    y_train_onehot = np_utils.to_categorical(y_train_label)
    y_test_onehot = np_utils.to_categorical(y_test_label)
    
    return(x_train_normalize, 
           x_test_normalize, 
           y_train_onehot, 
           y_test_onehot, 
           x_test_image, 
           y_test_label)


def create_model():
    model = Sequential()
    
    # convolutional layer 1
    model.add(Conv2D(filters=16,
                     kernel_size=(5,5),
                     padding='same',
                     input_shape=(28,28,1), 
                     activation='relu'))
    # pooling layer 1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # convolutional layer 2
    model.add(Conv2D(filters=36,
                     kernel_size=(5,5),
                     padding='same',
                     activation='relu'))
    # pooling layer 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # flat layer
    model.add(Flatten())    #36*7*7 = 1764 neurons
    # hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
    
    
def train_model(model, train_que, train_ans):
    train_history = model.fit(x=train_que, 
                              y=train_ans, 
                              validation_split=0.2, 
                              epochs=5, batch_size=500, verbose=2)
                              #epochs=10, batch_size=300
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')
    
    return
    
    
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
    print('Saved to {0}'.format(model_name))

    del model
    return
    

def import_model(model_name):
    model = load_model(model_name)
    print(model.summary())
    print('Loaded from {0}'.format(model_name))

    return model


def plot_images_labels_prediction(images, labels, prediction, idx=0, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25 
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label = " + str(labels[idx])
        if len(prediction) > 0:
            title += ", predict = " + str(prediction[idx]) 
            
        ax.set_title(title, fontsize=10) 
        ax.set_xticks([])
        ax.set_yticks([])        
        idx += 1 
    
    plt.show()
    return


def create_confusion_matrix(y_test_label, prediction):
    confusion_matrix = pd.crosstab(y_test_label, prediction, 
                                   rownames=['label'], colnames=['predict'])
    print(confusion_matrix)
    return
    
### ========== ###

name = r"C:\PyCodes\Keras\File\H_Keras_mnist_CNN 8.1.h5"
def main():
    already_trained = False
    (x_train_normalize, 
     x_test_normalize, 
     y_train_onehot, 
     y_test_onehot, 
     x_test_image, 
     y_test_label) = preprocess_data()
    
    if already_trained == False:
        model = create_model()
        train_history = train_model(model, x_train_normalize, y_train_onehot)
        # loss: 0.0838 - acc: 0.9746 - val_loss: 0.0469 - val_acc: 0.9859 - 56sec/epoch
        save_model(model, name)
        
    elif already_trained == True:
        model = import_model(name)

    evaluate_accuracy(model, x_test_normalize, y_test_onehot)
    prediction = model.predict_classes(x_test_normalize)
    plot_images_labels_prediction(x_test_image, y_test_label, prediction)
    create_confusion_matrix(y_test_label, prediction)
    
    return
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    