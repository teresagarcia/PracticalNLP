import re
from pprint import pprint

##NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
stop_words_nltk = set(stopwords.words('english'))

##SPACY 
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
spacy_model = spacy.load('en_core_web_sm')

#This will be our corpus which we will work on
corpus_original = "Need to finalize the demo corpus which will be used for this notebook and it should be done soon !!. It should be done by the ending of this month. But will it? This notebook has been run 4 times !!"
corpus = "Need to finalize the demo corpus which will be used for this notebook & should be done soon !!. It should be done by the ending of this month. But will it? This notebook has been run 4 times !!"

#lower case the corpus
corpus = corpus.lower()
print("Cambiar a minúsculas:\n", corpus)

#removing digits in the corpus
corpus = re.sub(r'\d+','', corpus)
print("Eliminar dígitos:\n",corpus)

#removing punctuations
import string
corpus = corpus.translate(str.maketrans('', '', string.punctuation))
print("Eliminar puntuación:\n",corpus)

#removing trailing whitespaces
corpus = ' '.join([token for token in corpus.split()])
print("Eliminar espacios en blanco innecesarios\n", corpus)


#Tokenizar el texto


tokenized_corpus_nltk = word_tokenize(corpus)
print("\nNLTK\nTokenized corpus:",tokenized_corpus_nltk)
tokenized_corpus_without_sw_nltk = [i for i in tokenized_corpus_nltk if not i in stop_words_nltk]
print("Tokenized corpus without stopwords:",tokenized_corpus_without_sw_nltk)

stopwords_spacy = spacy_model.Defaults.stop_words
nlp_spacy = spacy_model(corpus)
print("\nSpacy:")
tokenized_corpus_spacy = [token.text for token in nlp_spacy]
print("Tokenized Corpus:",tokenized_corpus_spacy)
tokenized_corpus_without_sw_spacy= [word for word in tokenized_corpus_spacy if not word in stopwords_spacy]

print("Tokenized corpus without stopwords",tokenized_corpus_without_sw_spacy)


print("Difference between NLTK and spaCy output:\n",
      set(tokenized_corpus_without_sw_nltk)-set(tokenized_corpus_without_sw_spacy))

#stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()

print("NLTK only:\n")
print("Before Stemming:")
print(corpus)

print("After Stemming:")
for word in tokenized_corpus_nltk:
    print(stemmer.stem(word),end=" ")


#Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()

print("Lemmatizer NLTK:\n", end="\n")
for word in tokenized_corpus_nltk:
    print(lemmatizer.lemmatize(word),end=" ")

print("Lemmatizer spacy:\n")
for token in nlp_spacy:
    print(token.lemma_, end=" ")

#POS tagging
#POS tagging using spacy
print("POS Tagging using spacy:")
doc = spacy_model(corpus_original)
# Token and Tag
for token in doc:
    print(token,":", token.pos_)

#pos tagging using nltk
nltk.download('averaged_perceptron_tagger')
print("POS Tagging using NLTK:")
pprint(nltk.pos_tag(word_tokenize(corpus_original)))

