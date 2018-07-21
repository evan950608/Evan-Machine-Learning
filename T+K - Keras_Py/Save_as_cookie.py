# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:16:35 2018

@author: evan9
"""

import shelve
import pickle

def save_shelve_value(key, value):
    obj = shelve.open('cache')
    obj[key] = value
    obj.close()

def get_shelve_value(key):    
    try:
        return shelve.open('cache')[key]
    except:
        return None

def save_pickle(fullname, value):
    with open(fullname, 'wb') as fileobj:
        pickle.dump(value, fileobj) #, protocol=pickle.HIGHEST_PROTOCOL

def load_pickle(fullname):
    with open(fullname, 'rb') as fileobj:
        value = pickle.load(fileobj)    
    return value


def main():
    #r = get_shelve_value('data')
    #save_shelve_value('data' , 'test123')
    #save_pickle('test.pkl',[1,2,3,])
    #r = load_pickle('test.pkl')
    print(r)

if __name__ == '__main__':
    main()
    pass