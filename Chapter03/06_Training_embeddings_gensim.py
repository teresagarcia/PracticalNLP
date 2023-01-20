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

#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.vocab)
print(words)

#Acess vector for one word
print(model_cbow['dog'])

