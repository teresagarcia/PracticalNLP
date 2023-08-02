#ya no existe

from gensim.summarization import summarize,summarize_corpus
from gensim.summarization.textcleaner import split_sentences
from gensim import corpora
import os
text = open(f"{os.getcwd()}/Chapter05/nlphistory.txt").read()

#summarize method extracts the most relevant sentences in a text
print("Summarize:\n",summarize(text, word_count=200, ratio = 0.1))


#the summarize_corpus selects the most important documents in a corpus:
sentences = split_sentences(text)# Creates a corpus where each document is a sentence.
tokens = [sentence.split() for sentence in sentences]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

# Extracts the most important documents (shown here in BoW representation)
print("-"*30,"\nSummarize Corpus\n",summarize_corpus(corpus,ratio=0.1))

