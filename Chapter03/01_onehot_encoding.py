documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs

print("Build the vocabulary:\n")
vocab = {}
count = 0
for doc in processed_docs:
    for word in doc.split():
        if word not in vocab:
            count = count +1
            vocab[word] = count
print(vocab)

#Get one hot representation for any string based on this vocabulary. 
#If the word exists in the vocabulary, its representation is returned. 
#If not, a list of zeroes is returned for that word. 
def get_onehot_vector(somestring):
    onehot_encoded = []
    for word in somestring.split():
        temp = [0]*len(vocab)
        if word in vocab:
            temp[vocab[word]-1] = 1 # -1 is to take care of the fact indexing in array starts from 0 and not 1
        onehot_encoded.append(temp)
    return onehot_encoded


print("Doc1:\n", processed_docs[1])
doc1_onehot = get_onehot_vector(processed_docs[1]) #one hot representation for a text from our corpus.
print("Doc1 one hot encoding:\n", doc1_onehot)

print("Example sentence: man and dog are good:")
print(get_onehot_vector("man and dog are good"))
#one hot representation for a random text, using the above vocabulary

print("Example sentence 2: man and man are good:")
print(get_onehot_vector("man and man are good"))

#Scikit learn
S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'

print("Scikit-learn:")
print(f"""Frases: \n {S1} \n {S2} \n {S3} \n {S4}""")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0]+data[1]+data[2]+data[3]
print("The data: ",values)

#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)

#One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n",onehot_encoded)