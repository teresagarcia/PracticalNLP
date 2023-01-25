import warnings
warnings.filterwarnings('ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pprint import pprint
import nltk
nltk.download('punkt')

data = ["dog bites man",
        "man bites dog",
        "dog eats meat",
        "man eats food"]

tagged_data = [TaggedDocument(words=word_tokenize(word.lower()), tags=[str(i)]) for i, word in enumerate(data)]
print("tagged data:\n", tagged_data)

#dbow
print("~~~ DBOW ~~~")
model_dbow = Doc2Vec(tagged_data,vector_size=20, min_count=1, epochs=2,dm=0)

print(model_dbow.infer_vector(['man','eats','food']))#feature vector of man eats food

print("Top 5 most similar words for 'man'")
print(model_dbow.wv.most_similar("man",topn=5))#top 5 most simlar words.

print("Similarity between 'dog' and 'man'")
print(model_dbow.wv.n_similarity(["dog"],["man"]))

print("~~~ DM ~~~")
#dm
model_dm = Doc2Vec(tagged_data, min_count=1, vector_size=20, epochs=2,dm=1)

print("Inference Vector of man eats food\n ",model_dm.infer_vector(['man','eats','food']))

print("Most similar words to man in our corpus\n",model_dm.wv.most_similar("man",topn=5))
print("Similarity between man and dog: ",model_dm.wv.n_similarity(["dog"],["man"]))

print("Similarity between 'covid' (out of vocabulary) and 'man: '", model_dm.wv.n_similarity(['covid'],['man']))