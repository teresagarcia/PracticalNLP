import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from pprint import pprint

#tokenize, remove stopwords, non-alphabetic words, lowercase
def preprocess(textstring):
   stops =  set(stopwords.words('english'))
   tokens = word_tokenize(textstring)
   return [token.lower() for token in tokens if token.isalpha() and token not in stops]

# This is a sample path of your downloaded data set. This is currently set to a windows based path . 
# Please update it to your actual download path regradless of your choice of operating system 

data_path = os.path.join(f"{os.getcwd()}/data/Chapter07/booksummaries.txt")

summaries = []
for line in open(data_path, encoding="utf-8"):
   temp = line.split("\t")
   summaries.append(preprocess(temp[6]))

# Create a dictionary representation of the documents.

dictionary = Dictionary(summaries)

# Filter infrequent or too frequent words.

dictionary.filter_extremes(no_below=10, no_above=0.5)
corpus = [dictionary.doc2bow(summary) for summary in summaries]

# Make a index to word dictionary.

temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

#Train the topic model

model = LdaModel(corpus=corpus, id2word=id2word,iterations=400, num_topics=10)
top_topics = list(model.top_topics(corpus))
pprint("~~~~ Top Topics ~~~~~~")
pprint(top_topics)

for idx in range(10):
    print("Topic #%s:" % idx, model.print_topic(idx, 10))
print("=" * 20)

pprint("~~~~~~~ Latent Semantic Indexing/LSA ~~~~~")
from gensim.models import LsiModel
lsamodel = LsiModel(corpus, num_topics=10, id2word = id2word)  # train model

pprint(lsamodel.print_topics(num_topics=10, num_words=10))

for idx in range(10):
    print("Topic #%s:" % idx, lsamodel.print_topic(idx, 10))
 
print("=" * 20)