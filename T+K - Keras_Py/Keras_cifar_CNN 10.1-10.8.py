# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:38:51 2018

@author: evan9
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import inspect

### SET UP ###
def preprocess_data():
    np.random.seed(10)
    (x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
#    print('''
#train data:
#    images: {0}
#    labels: {1}
#test data:
#    images: {2}
#    labels: {3}
#'''.format(x_img_train.shape, y_label_train.shape, x_img_test.shape, y_label_test.shape))
    
    x_img_train_normalize = x_img_train.astype('float32') / 255
    x_img_test_normalize = x_img_test.astype('float32') / 255
    y_label_train_onehot = np_utils.to_categorical(y_label_train)
    y_label_test_onehot = np_utils.to_categorical(y_label_test)
    
    return(x_img_train_normalize, 
           x_img_test_normalize, 
           y_label_train_onehot, 
           y_label_test_onehot, 
           x_img_test, 
           y_label_test)
    
    
def create_model():
    model = Sequential()
    
    # convolutional layer 1
    model.add(Conv2D(filters=32, kernel_size=(3,3), 
                     input_shape=(32,32,3), 
                     activation='relu', 
                     padding='same'))
    model.add(Dropout(rate=0.25))
    # pooling layer 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # convolutional layer 2
    model.add(Conv2D(filters=64, kernel_size=(3,3), 
                     activation='relu', 
                     padding='same'))
    model.add(Dropout(rate=0.25))
    # pooling layer 2
    model.add(MaxPooling2D(pool_size=(2,2)))
    # flat layer
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    # hidden layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    # output layer
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
    

def cifar_image_categories():
    label_dict = {0: "airplane", 
                  1: "automobile", 
                  2: "bird", 
                  3: "cat", 
                  4: "deer", 
                  5: "dog", 
                  6: "frog", 
                  7: "horse", 
                  8: "ship", 
                  9: "truck"}
    return label_dict

### TRAINING ###
def train_model(model, train_que, train_ans):
    train_history = model.fit(x=train_que, 
                              y=train_ans, 
                              validation_split=0.2, 
                              epochs=10, batch_size=128, verbose=2)
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
    print()
    print('accuracy= %.4f' % (scores[1],))
    
    return


def make_prediction(model, test_que):
    prediction = model.predict_classes(test_que)
    predicted_probability = model.predict(test_que)
    
    return prediction, predicted_probability


def plot_images_labels_prediction(images, labels, prediction, idx=0, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25 
    label_dict = cifar_image_categories()
    
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx + i], cmap='binary')
        
        title = str(idx + i) + ',' + label_dict[labels[idx + i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idx + i]] 
            
        ax.set_title(title, fontsize=10) 
        ax.set_xticks([])
        ax.set_yticks([])        
    
    plt.show()
    return


def show_predicted_probability(test_que, test_ans, prediction, predicted_probability, idx=0):
    label_dict = cifar_image_categories()
    print('label:', label_dict[test_ans[idx][0]], ';'
          'predict:', label_dict[prediction[idx]])
    
    if label_dict[test_ans[idx][0]] == label_dict[prediction[idx]]:
        print('Prediction successful')
    else:
        print('Prediction failed')
    
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(test_que[idx], (32,32,3)))
    plt.show()
    
    for j in range(10):
        print(label_dict[j] + ' Probability:%1.9f' % (predicted_probability[idx][j]))


def create_confusion_matrix(y_label_test, prediction):
    print(cifar_image_categories())
    confusion_matrix = pd.crosstab(y_label_test.reshape(-1), prediction, 
                                   rownames=['label'], colnames=['predict'])
    print(confusion_matrix)
    return


### ========== ###

def main():
    global local_vars
    already_trained = True
    
    (x_img_train_normalize, 
     x_img_test_normalize, 
     y_label_train_onehot, 
     y_label_test_onehot, 
     x_img_test, 
     y_label_test) = preprocess_data()
    
    if already_trained == False:
        model = create_model()
        train_history = train_model(model, x_img_train_normalize, y_label_train_onehot)
        save_model(model, name)
    elif already_trained == True:
        model = import_model(name)
        
    evaluate_accuracy(model, x_img_test_normalize, y_label_test_onehot)
    
    (prediction, predicted_probability) = make_prediction(model, x_img_test_normalize)
    show_predicted_probability(x_img_test, y_label_test, 
                               prediction, predicted_probability, idx=102)
    
    plot_images_labels_prediction(x_img_test, y_label_test, prediction, idx=100)
    create_confusion_matrix(y_label_test, prediction)
    
    local_vars = inspect.currentframe().f_locals
            
    return local_vars


if __name__ == '__main__':
    name = r"C:\PyCodes\Keras\File\H_Keras_mnist_CNN 10.1-10.8.h5"
    local_vars = {}
    local_vars_detail = {}
    
    local_vars = main()
    
    for key,value in sorted(local_vars.items()):
        if type(value) == np.ndarray:
            local_vars_detail[key] = value.shape
        else:
            local_vars_detail[key] = [type(value), value]
            
#    input('Press enter to continue...')




