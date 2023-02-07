import pandas as pd
import wget
import os
from zipfile import ZipFile


import fastai
from fastai import *
from fastai.text import *
from fastai.text.data import LMDataLoader
import pandas as pd
import numpy as np
from functools import partial
import io

path = os.getcwd()+'/data/Chapter04'
if not os.path.exists(path + '/SMSSpamCollection'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    wget.download(url,path)
    temp=path+'/smsspamcollection.zip'          
    file = ZipFile(temp)           
    file.extractall(path)          
    file.close()

df = pd.read_csv(path + '/SMSSpamCollection', sep='\t',  header=None, names=['target', 'text'])

print(df.head())

print("Number of rows and columns, class distribution")
print(df.shape) #Number of rows (instances) and columns in the dataset
print(df["target"].value_counts()/df.shape[0]) #Class distribution in the dataset

from sklearn.model_selection import train_test_split

# split data into training and validation set
df_train, df_test = train_test_split(df,stratify = df['target'], test_size = 0.2, random_state = 2020)


###Incomplete....