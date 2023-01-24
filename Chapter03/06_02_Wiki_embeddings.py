from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import time

file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p11659683p13159682.bz2"
#Preparing the Training data
wiki = WikiCorpus(file_name, dictionary={})
sentences = list(wiki.get_texts())

#if you get a memory error executing the lines above
#comment the lines out and uncomment the lines below. 
#loading will be slower, but stable.
# wiki = WikiCorpus(file_name, processes=1, dictionary={})
# sentences = list(wiki.get_texts())

#if you still get a memory error, try settings processes to 1 or 2 and then run it again.

#CBOW
print("~~~CBOW with Word2Vec~~~")
start = time.time()
word2vec_cbow = Word2Vec(sentences,min_count=10, sg=0)
end = time.time()

print("CBOW Model Training Complete.\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))



print("Summarize the loaded model")
print(word2vec_cbow)
print("-"*30)

print("Summarize vocabulary")
words = list(word2vec_cbow.wv.key_to_index)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_cbow.wv['film'])}")
print("Vector for 'film'")
print(word2vec_cbow.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_cbow.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_cbow.wv.similarity('film', 'tiger'))
print("-"*30)



# save model
from gensim.models import Word2Vec, KeyedVectors   
word2vec_cbow.wv.save_word2vec_format('models/word2vec_cbow.bin', binary=True)

# load model
# new_modelword2vec_cbow = Word2Vec.load('word2vec_cbow.bin')
# print(word2vec_cbow)


#SkipGram
print("~~~Skip Gram with Word2Vec~~~")
start = time.time()
word2vec_skipgram = Word2Vec(sentences,min_count=10, sg=1)
end = time.time()

print("SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))

print("Summarize the loaded model")
print(word2vec_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_skipgram.wv.key_to_index)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_skipgram.wv['film'])}")
print(word2vec_skipgram.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_skipgram.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_skipgram.wv.similarity('film', 'tiger'))
print("-"*30)

# save model
word2vec_skipgram.wv.save_word2vec_format('models/word2vec_sg.bin', binary=True)

# load model
# new_model_skipgram = Word2Vec.load('model_skipgram.bin')
# print(model_skipgram)

print("~~~FastText~~~")


#CBOW
print("~~~CBOW with FastText~~~")
start = time.time()
fasttext_cbow = FastText(sentences, sg=0, min_count=10)
end = time.time()

print("FastText CBOW Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))

print("Summarize the loaded model")
print(fasttext_cbow)
print("-"*30)

#Summarize vocabulary
words = list(fasttext_cbow.wv.key_to_index)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(fasttext_cbow.wv['film'])}")
print("Vector for 'film'")
print(fasttext_cbow.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",fasttext_cbow.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",fasttext_cbow.wv.similarity('film', 'tiger'))
print("-"*30)

#SkipGram
print("~~~Skip gram with FastText~~~")
start = time.time()
fasttext_skipgram = FastText(sentences, sg=1, min_count=10)
end = time.time()

print("FastText SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))

print("Summarize the loaded model")
print(fasttext_skipgram)
print("-"*30)

print("Summarize vocabulary")
words = list(fasttext_skipgram.wv.key_to_index)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(fasttext_skipgram.wv['film'])}")
print("Vector for 'film'")
print(fasttext_skipgram.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",fasttext_skipgram.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",fasttext_skipgram.wv.similarity('film', 'tiger'))
print("-"*30)