''' 
author:Arnab Tarafder
date: 03/02/2021, 12:20 am
'''

import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing

def preprocess(file_location):
    '''
    input: 
        file_location: location of the .dat file
    output:
        dataframe cotaining cleaned data
    '''
    d=[]
    ''' here getting rid of extra descriptions line that all start with @
        see the .dat file to understand details
    '''
    
    for i in open(file_location).readlines():
        if i[0]!='@':
            d.append(i.rstrip('\n').split(','))
    
    data=pd.DataFrame(d, columns=[str(i) for i in range(len(d[0]))])
    
    ''' making string values numeric, except for the class type, initially all were Object type'''
    
    for i in range(len(d[0])-1):
        data[str(i)]=data[str(i)].astype('float64')
    
    le=preprocessing.LabelEncoder()
	
    data[str(len(d[0])-1)]=le.fit_transform(data[str(len(d[0])-1)])
	
    return data

def make_sequence(train):
    
    '''
    input: takes DataFrame 
    ouput: returns majority sequence and minority sequence in numpy array, shape(time_steps, batch_size, features)
    '''
    
    maj=np.array(train[train[:]['8']==' negative'])
    minority=np.array(train[train[:]['8']==' positive'])
    
    batch_size=minority.shape[1]
    
    train_majority=maj[:,:maj.shape[1]-1]
    train_minority=minority[:,:minority.shape[1]-1]
    
    train_minority_seq=np.zeros((1,train_minority.shape[0],train_minority.shape[1]))
    train_minority_seq[0]=train_minority
    
    maj_seq_len=maj.shape[0]//minority.shape[0]
    
    maj_seq=np.zeros((maj_seq_len,batch_size,train_majority.shape[1]))
    
    for i in range(0,maj_seq_len,1):
        maj_seq[i]=train_majority[i:i+batch_size]
    
    
    
    return torch.from_numpy(maj_seq).float(), torch.from_numpy(train_minority_seq).float()
	
	
	