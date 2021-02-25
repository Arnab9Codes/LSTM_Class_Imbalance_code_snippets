''' 
author:Arnab Tarafder
date: 14/02/2021, 7:00 pm
'''

import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing
import re

def clean(file_location):   
    '''
        input: takes file_location
        output: dataframe of cleaned data
    '''
    
    d=[]
       
    for i in open(file_location).readlines():
        if i[0]!='@':
            d.append(i.rstrip('\n').split(','))

    types={} # holds the datatypes required

    for i in range(len(d[0])):
        if re.search('[0-9]',d[0][i]):

            types[str(i)]=float
        else:
            types[str(i)]=str


    data=pd.DataFrame(d, columns=[str(i) for i in range(len(d[0]))])
    data=data.astype(types)
    
    # label encoding str type columns

    le=preprocessing.LabelEncoder() 

    for i in range(len(data.columns)):
        if isinstance(data[str(i)][0],str):
            data[str(i)]=le.fit_transform(data[str(i)])
    
    return data


	
def seq_gen(data, mj=0, mn=1):
    '''
    input: data of DataFrame, mj: majority class label, mn: minority class label
    
    output: numpy tensor of shape (timsteps, 2*min_sample_size, features)
            at each timestep, majority class and minority class samples distribution is equal
    '''
    data_mj=data[data[str(len(data.columns)-1)]==mj]
    data_mj=np.array(data_mj)
    data_mn=data[data[str(len(data.columns)-1)]==mn]
    data_mn=np.array(data_mn)
    
    '''seq_len is the number of timesteps.'''
	
    seq_len=data_mj.shape[0]//data_mn.shape[0]
	
    '''batch_size = number of minority samples'''
	
    batch_size=data_mn.shape[0]
    
    seq=np.zeros((seq_len, data_mn.shape[0]*2, data_mn.shape[1]))
    #print(seq.shape)
    
    for i in range(seq_len):
        seq[i,0:batch_size]=data_mj[(i*batch_size):( (i*batch_size)+batch_size)]
        seq[i,batch_size:2*batch_size]=data_mn
        
    return seq#torch.from_numpy(seq).float()


def get_Xy(data):
    
    '''
    input: data:numpy or torch tensor
    output: X(predictors), y(target class)
    '''
    trainx=data[:,:,:-1]
    trainy=data[:,:,-1]
    
    return trainx,trainy
	
	
