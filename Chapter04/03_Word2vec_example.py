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

path_to_model = 'models/GoogleNews-vectors-negative300.bin'
training_data_path = "data/Chapter04/sentiment_sentences.txt"

#Load W2V model. This will take some time. 
print("Loading Google News Word2Vec model...")
w2v_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
print('done loading Word2Vec')

#Read text data, cats.
#the file path consists of tab separated sentences and cats.
print("Reading text data and cats, divide them in 2 arrays")
texts = []
cats = []
fh = open(training_data_path)
for line in fh:
    text, sentiment = line.split("\t")
    texts.append(text)
    cats.append(sentiment)

#Inspect the model
print("Inspecting the model...")
word2vec_vocab = w2v_model.key_to_index.keys()
word2vec_vocab_lower = [item.lower() for item in word2vec_vocab]
print("Vocabulary length: ", len(word2vec_vocab))


#Inspect the dataset
print("Inspecting the dataset... Example text and its category:")
print(len(cats), len(texts))
print(texts[1])
print(cats[1])

#preprocess the text.
print("Preprocessing the text: removing stopwords, digits and lowercase tokens.")
def preprocess_corpus(texts):
    mystopwords = set(stopwords.words("english"))
    def remove_stops_digits(tokens):
        #Nested function that lowercases, removes stopwords and digits from a list of tokens
        return [token.lower() for token in tokens if token.lower() not in mystopwords and not token.isdigit()
               and token not in punctuation]
    #This return statement below uses the above function to process twitter tokenizer output further. 
    return [remove_stops_digits(word_tokenize(text)) for text in texts]

texts_processed = preprocess_corpus(texts)
print("Length of cats and processed texts arrays:")
print(len(cats), len(texts_processed))
print("Example text and category after preprocessing:")
print(texts_processed[1])
print(cats[1])



print("Creating a feature vector by averaging all embeddings for all sentences")
def embedding_feats(list_of_lists):
    DIMENSION = 300
    zero_vector = np.zeros(DIMENSION)
    feats = []
    for tokens in list_of_lists:
        feat_for_this =  np.zeros(DIMENSION)
        count_for_this = 0 + 1e-5 # to avoid divide-by-zero 
        for token in tokens:
            if token in w2v_model:
                feat_for_this += w2v_model[token]
                count_for_this +=1
        if(count_for_this!=0):
            feats.append(feat_for_this/count_for_this) 
        else:
            feats.append(zero_vector)
    return feats


train_vectors = embedding_feats(texts_processed)
print("Length of train vectors")
print(len(train_vectors))
print("Si lo he entendido es 1 vector de dim 300 para cada frase con la media de todas sus palabras")
print(len(train_vectors[0]))


print("Take any classifier (LogisticRegression here), and train/test it like before.")
classifier = LogisticRegression(random_state=1234)
train_data, test_data, train_cats, test_cats = train_test_split(train_vectors, cats)
classifier.fit(train_data, train_cats)
print("Accuracy: ", classifier.score(test_data, test_cats))
preds = classifier.predict(test_data)
print(classification_report(test_cats, preds))