#titanic_01.py
import os,math
import keras
import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten,Conv2D,MaxPooling2D
from sklearn import preprocessing

def PreprocessData(raw_df):
    df=raw_df.drop(['name'], axis=1)
    df['age'] = df['age'].fillna(df['age'].mean())
    df['fare'] = df['fare'].fillna(df['fare'].mean())
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])
#    print('---------------------')
#    print(x_OneHot_df[:2])
    
    ndarray = x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

def preparing_trained_data():
    filepath = os.path.join(r'C:\Users\Ennio\.keras\datasets','titanic3.xls')
    global all_df 
    all_df = pd.read_excel(filepath)
    #print(all_df[:2])
    cols=['survived','name','pclass' ,'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']        
    all_df=all_df[cols]
    
    Jack = pd.Series([0 ,'Jack',3, 'male'  , 23, 1, 0,  5.0000,'S'])
    Rose = pd.Series([1 ,'Rose',1, 'female', 20, 1, 0, 100.0000,'S'])
    JR_df = pd.DataFrame([list(Jack),list(Rose)],  
                      columns=['survived', 'name','pclass', 'sex', 
                       'age', 'sibsp','parch', 'fare','embarked'])
    all_df=pd.concat([all_df,JR_df])
    
    
    msk = np.random.rand(len(all_df)) < 0.8
    train_df = all_df[msk]
    test_df = all_df[~msk]        
    #train_df.to_csv('aaa.csv')

    x_train_normalize,y_train_onehot=PreprocessData(train_df)
    x_test_normalize,y_test_onehot=PreprocessData(test_df)
#    print('##################')        
#    print(x_train_normalize[:3])
#    print('##################')                
#    print(y_train_onehot[:3])              
    
    return x_train_normalize, y_train_onehot, x_test_normalize, y_test_onehot
    

def creating_model():
    model = keras.models.Sequential()    
    
    model.add(Dense(units=40, input_dim=9, 
                    kernel_initializer='uniform', 
                    activation='relu'))
    model.add(Dense(units=30, 
                    kernel_initializer='uniform', 
                    activation='relu'))
    model.add(Dense(units=1, 
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    print(model.summary())
    return model

def training_model(model, x_train, y_train):
    model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])    
    train_history=model.fit(x=x_train,
                            y=y_train,validation_split=0.1, 
                            epochs=30, batch_size=30,verbose=2)    
    return train_history

def evaluating_model(model,x_test, y_test):
    scores = model.evaluate(x_test, y_test)
    print('accuracy=',scores[1])    
    pass

def predicting_model(model, x_test):
    prediction=model.predict_classes(x_test)
    predicted_probability = model.predict(x_test)
    return prediction  , predicted_probability  
    
def saving_model(model , model_name):
    model.save(model_name)  
    print(model.summary())        
    del model  # deletes the existing model
    
def loading_model(model_name):
    model = keras.models.load_model(model_name)
    return model

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

model_name = 'SaveModel/my_model_taianic.h5'
def main():
    debug_def = '2'
    if debug_def == '':
        ### Preparing Data ###
        x_train, y_train , x_test, y_test = preparing_trained_data()
        print(x_train[0] , y_train[0])

        ### Createing Model ###        
        model = creating_model() 

        ### Training Data ###        
        train_history = training_model(model,x_train, y_train)
        show_train_history(train_history,'acc','val_acc')
        show_train_history(train_history,'loss','val_loss')

        ### Saving Data ###        
        saving_model(model,model_name)
    
        ### Evaluating Data ###        
        evaluating_model(model,x_test, y_test)
        return        

        pass
    elif debug_def == '2':
        ### Preparing Data ###
        x_train, y_train , x_test, y_test = preparing_trained_data()

        model = loading_model(model_name)        
        ### Predicting Data ###
        all_Features,Label=PreprocessData(all_df)
        all_probability=model.predict(all_Features)
        pd=all_df
        pd.insert(len(all_df.columns),
                  'probability',all_probability)
        print(pd[-2:])
        pass
    pass

if __name__ == '__main__':
    main()
    pass