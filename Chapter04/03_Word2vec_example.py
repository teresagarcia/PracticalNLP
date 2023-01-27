#basic imports
import warnings
warnings.filterwarnings('ignore')
import os
import wget
import gzip
import shutil
from time import time

#pre-processing imports
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

#imports related to modeling
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if not os.path.isfile("data/Chapter04/sentiment_sentences.txt"):
    yelp_file = open("data/Chapter04/yelp_labelled.txt", "r")
    yelp_text = yelp_file.read()
    amazon_file = open("data/Chapter04/amazon_cells_labelled.txt", "r")
    amazon_text = amazon_file.read()
    imdb_file = open("data/Chapter04/imdb_labelled.txt", "r")
    imdb_text = imdb_file.read()
    f = open("data/Chapter04/sentiment_sentences.txt", "a")
    full_text = yelp_text + amazon_text + imdb_text
    f.write(full_text)