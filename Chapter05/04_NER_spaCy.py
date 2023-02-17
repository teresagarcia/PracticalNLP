### from: https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
print(sent)

print("Our chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.")

pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)

print("Using IOB tags")

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)

print("Using spaCy...")

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])

from bs4 import BeautifulSoup
import requests
import re

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
    
ny_bb = url_to_string('https://kbizoom.com/jyjs-jaejoong-faces-the-risk-of-up-to-2-years-in-jail-or-a-fine-of-4-million-won-for-posting-this-photo-onto-his-sns/')
article = nlp(ny_bb)
print(len(article.ents))

print("Entities classified by labels")
labels = [x.label_ for x in article.ents]
print(Counter(labels))

print("3 most frequent tokens")
items = [x.text for x in article.ents]
print(Counter(items).most_common(3))

sentences = [x for x in article.sents]
print(sentences[13])

displacy.serve(nlp(str(sentences[13])), style='ent')

displacy.serve(nlp(str(sentences[13])), style='dep', options = {'distance': 120})

print("Verbatim, extract part-of-speech and lemmatize this sentence.")
print([(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[13])) if not y.is_stop and y.pos_ != 'PUNCT']])

print("Named entity extracition...")
print(dict([(str(x), x.label_) for x in nlp(str(sentences[13])).ents]))

print("IOB tags")
print(print([(x, x.ent_iob_, x.ent_type_) for x in sentences[13]]))