
import os
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Read the dataset’s README to understand the data format. 

data_path = f"{os.getcwd()}/data/Chapter07/booksummaries.txt"
mydata = {} #titles-summaries dictionary object
for line in open(data_path, encoding="utf-8"):
    temp = line.split("\t")
    mydata[temp[2]] = temp[6]

#prepare the data for doc2vec, build and save a doc2vec model
train_doc2vec = [TaggedDocument((word_tokenize(mydata[t])), tags=[t]) for t in mydata.keys()]
model = Doc2Vec(vector_size=50, alpha=0.025, min_count=10, dm =1, epochs=100)
model.build_vocab(train_doc2vec)
model.train(train_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
model.save(f"{os.getcwd()}/models/d2v.model")



#Use the model to look for similar texts
model = Doc2Vec.load(f"{os.getcwd()}/models/d2v.model")

#This is a sentence from the summary of “Animal Farm” on Wikipedia:
#https://en.wikipedia.org/wiki/Animal_Farm
sample = """
Napoleon enacts changes to the governance structure of the farm, replacing meetings with a committee of pigs who will run the farm.
 """
new_vector = model.infer_vector(word_tokenize(sample))
sims = model.docvecs.most_similar([new_vector])
print(sims)

