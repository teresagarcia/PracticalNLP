from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# define training data
#Genism word2vec requires that a format of ‘list of lists’ be provided for training where every document contained in a list.
#Every list contains lists of tokens of that document.
corpus = [['dog','bites','man'], ["man", "bites" ,"dog"],["dog","eats","meat"],["man", "eats","food"]]

#Training the model
model_cbow = Word2Vec(corpus, min_count=1,sg=0) #using CBOW Architecture for trainnig
model_skipgram = Word2Vec(corpus, min_count=1,sg=1) #using skipGram Architecture for training 

print("~~~CBOW~~~")
print("Summarize the loaded model")
print(model_cbow)

print("Summarize vocabulary")
words = list(model_cbow.wv.key_to_index)
print(words)

#Access vector for one word
print("Vector for 'dog'")
print(model_cbow.wv['dog'])

#Compute similarity 
print("Similarity between eats and bites:",model_cbow.wv.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_cbow.wv.similarity('eats', 'man'))

#Most similarity
print("Most similar words for 'meat'")
print(model_cbow.wv.most_similar('meat'))

# save model
model_cbow.save('Chapter03/model_cbow.bin')

# load model
new_model_cbow = Word2Vec.load('Chapter03/model_cbow.bin')
print("new loaded model:")
print(new_model_cbow)


print("-"*50)

print("~~~Skip gram~~~")

print("Summarize the loaded model")
print(model_skipgram)

print("Summarize vocabulary")
words = list(model_skipgram.wv.key_to_index)
print(words)

#Acess vector for one word
print("Vector for 'dog'")
print(model_skipgram.wv['dog'])

#Compute similarity 
print("Similarity between eats and bites:",model_skipgram.wv.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_skipgram.wv.similarity('eats', 'man'))

#Most similarity
print("Most similar words for 'meat'")
print(model_skipgram.wv.most_similar('meat'))

# save model
model_skipgram.save('model_skipgram.bin')

# load model
new_model_skipgram = Word2Vec.load('model_skipgram.bin')
print("new loaded model:")
print(new_model_skipgram)

