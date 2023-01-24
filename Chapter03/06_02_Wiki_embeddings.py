from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import time

file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p11659683p13159682.bz2"
#Preparing the Training data
# wiki = WikiCorpus(file_name, dictionary={})
# sentences = list(wiki.get_texts())

#if you get a memory error executing the lines above
#comment the lines out and uncomment the lines below. 
#loading will be slower, but stable.
wiki = WikiCorpus(file_name, processes=1, dictionary={})
sentences = list(wiki.get_texts())

#if you still get a memory error, try settings processes to 1 or 2 and then run it again.

