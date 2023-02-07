import os
from zipfile import ZipFile
import tarfile
from fastai.text.all import *
import pandas as pd

data_path = os.getcwd() + "/data/Chapter04"
reviews_zip_path = data_path + '/archive.zip'
reviews_tgz_path = data_path + "/amazon_review_polarity_csv.tgz"
amazon_reviews_path = data_path + "/amazon_review_polarity_csv"

if not os.path.exists(amazon_reviews_path):
    with ZipFile(reviews_zip_path) as file:          
        file.extractall(data_path)      
    with tarfile.open(reviews_tgz_path) as file:
        file.extractall(data_path)    

train_df = pd.read_csv(amazon_reviews_path + '/train.csv', names=['label', 'title', 'text'], nrows=40000)
valid_df = pd.read_csv(amazon_reviews_path + '/test.csv', names=['label', 'title', 'text'], nrows=2000)
print(train_df.head())

sample_text = train_df['text'][0]
print(sample_text)

import torch
import torchtext
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")

tokens = L(tokenizer(sample_text))
print(tokens)

from collections import Counter

token_counter = Counter()

for text in train_df['text']:
    tokens = tokenizer(text)
    token_counter.update(tokens)

print("25 most common tokens")
print(token_counter.most_common(n=25))

print("25 least common tokens")
print(token_counter.most_common()[-25:])