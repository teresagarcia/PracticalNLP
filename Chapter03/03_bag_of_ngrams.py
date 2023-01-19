from sklearn.feature_extraction.text import CountVectorizer

#our corpus
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]

processed_docs = [doc.lower().replace(".","") for doc in documents]
print(processed_docs)

#Ngram vectorization example with count vectorizer and uni, bi, trigrams
count_vect = CountVectorizer(ngram_range=(1,3))

#Build a BoNgrams representation for the corpus
bo_ngrams_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BoNgrams rep for first 2 documents
print("BoNgrams representation for 'dog bites man': ", bo_ngrams_rep[0].toarray())
print("BoNgrams representation for 'man bites dog: ",bo_ngrams_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp1 = count_vect.transform(["dog and dog are friends"])
temp2 = count_vect.transform(["dog and dog are friends and man bites dog", "my dog eats man"])

print("BoNgrams representation for 'dog and dog are friends':", temp1.toarray())
print("BoNgrams representation for 'dog and dog are friends and man bites dog', 'my dog eats man':", temp2.toarray())