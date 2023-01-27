import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import wget
import os.path

#Descargar y cargar los datos
print("~~~Section 1: Load and explore the dataset~~~")
train_url = 'https://raw.githubusercontent.com/practical-nlp/practical-nlp/master/Ch4/Data/Sentiment and Emotion in Text/train_data.csv'
train_path = 'data/Chapter04/Sentiment_Emotion_Text_Train.csv'
if not os.path.isfile(train_path):
    wget.download(train_url, out = train_path)

test_url = 'https://raw.githubusercontent.com/practical-nlp/practical-nlp/master/Ch4/Data/Sentiment and Emotion in Text/test_data.csv'
test_path = 'data/Chapter04/Sentiment_Emotion_Text_Test.csv'
if not os.path.isfile(test_path):
    wget.download(test_url, out = test_path)

df = pd.read_csv(train_path)
print("~~~ Shape and preview of data ~~~")
print(df.shape)
print(df.head())

print("~~~ Count by sentiment~~~")
print(df['sentiment'].value_counts())


#Let us take the top 3 categories and leave out the rest.
print("~~~ Create subset with 'neutral', 'happiness' and 'worry' categories ~~~")
shortlist = ['neutral', "happiness", "worry"]
df_subset = df[df['sentiment'].isin(shortlist)]
print("Shape of the subset: ", df_subset.shape)


print("~~~ Text prepocessing ~~~")

#strip_handles removes personal information such as twitter handles, which don't
#contribute to emotion in the tweet. preserve_case=False converts everything to lowercase.
tweeter = TweetTokenizer(strip_handles=True,preserve_case=False)
mystopwords = set(stopwords.words("english"))

print("Tokenize tweets, remove stopwords and numbers.")
#Function to tokenize tweets, remove stopwords and numbers. 
#Keeping punctuations and emoticon symbols could be relevant for this task!
def preprocess_corpus(texts):
    def remove_stops_digits(tokens):
        #Nested function that removes stopwords and digits from a list of tokens
        return [token for token in tokens if token not in mystopwords and not token.isdigit()]
    #This return statement below uses the above function to process twitter tokenizer output further. 
    return [remove_stops_digits(tweeter.tokenize(content)) for content in texts]

#df_subset contains only the three categories we chose. 
mydata = preprocess_corpus(df_subset['content'])
mycats = df_subset['sentiment']
print("Length of data and categories subset:")
print(len(mydata), len(mycats))


print("Split data into train and test, prepare training data in doc2vec format, train doc2vec model")
#Split data into train and test, following the usual process
train_data, test_data, train_cats, test_cats = train_test_split(mydata,mycats,random_state=1234)

#prepare training data in doc2vec format:
train_doc2vec = [TaggedDocument((d), tags=[str(i)]) for i, d in enumerate(train_data)]
#Train a doc2vec model to learn tweet representations. Use only training data!!
model = Doc2Vec(vector_size=50, alpha=0.025, min_count=5, dm=1, epochs=100)
model.build_vocab(train_doc2vec)
model.train(train_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
model.save("models/d2v.model")
print("Model Saved")


#Infer the feature representation for training and test data using the trained model
model = Doc2Vec.load("models/d2v.model")
#infer in multiple steps (epochs) to get a stable representation. 
train_vectors = [model.infer_vector(list_of_tokens, epochs=50) for list_of_tokens in train_data]
test_vectors = [model.infer_vector(list_of_tokens, epochs=50) for list_of_tokens in test_data]

#Use any regular classifier like logistic regression
from sklearn.linear_model import LogisticRegression
print("Classify data")
myclass = LogisticRegression(class_weight="balanced") #because classes are not balanced. 
myclass.fit(train_vectors, train_cats)

preds = myclass.predict(test_vectors)
from sklearn.metrics import classification_report, confusion_matrix
print("Report of results:")
print(classification_report(test_cats, preds))

print("Confusion matrix:")
print(confusion_matrix(test_cats,preds))

