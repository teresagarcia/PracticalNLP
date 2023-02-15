import spacy
import textacy
import textacy.extract.keyterms as ke
from textacy import *

#Load a spacy model, which will be used for all further processing.
en = textacy.load_spacy_lang("en_core_web_sm")

#Let us use a sample text file, nlphistory.txt, which is the text from the history section of Wikipedia's
#page on Natural Language Processing 
#https://en.wikipedia.org/wiki/Natural_language_processing

mytext = open('data/Chapter05/agriculture.txt').read()

#convert the text into a spacy document.
doc = textacy.make_spacy_doc(mytext, lang=en)

print("Extract key terms from a document using the TextRank algorithm")
print(ke.textrank(doc, topn=5))

print("Print the keywords using TextRank algorithm, as implemented in Textacy.")
print("Textrank output: ", [kps for kps, weights in ke.textrank(doc, normalize="lemma", topn=5)])
print("Print the key words and phrases, using SGRank algorithm, as implemented in Textacy")
print("SGRank output: ", [kps for kps, weights in ke.sgrank(doc, topn=5)])

#To address the issue of overlapping key phrases, textacy has a function: aggregage_term_variants.
#Choosing one of the grouped terms per item will give us a list of non-overlapping key phrases!
print("Get a list of non-overlapping key phrases by using the function aggregage_term_variants")
terms = set([term for term,weight in ke.sgrank(doc)])
print(textacy.extract.utils.aggregate_term_variants(terms))

#A way to look at key phrases is just consider all noun chunks as potential ones. 
#However, keep in mind this will result in a lot of phrases, and no way to rank them!
print("Print all noun chunks and look at key phrases")
print([chunk for chunk in textacy.extract.noun_chunks(doc)])

