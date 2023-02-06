import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd #to work with csv files

#import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from sklearn.metrics import accuracy_score

#pre-processing of text
import string
import re

#import classifiers from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('WebAgg')
our_data = pd.read_csv(os.getcwd()+"/data/Chapter04/Full-Economic-News-DFE-839861.csv", encoding = "ISO-8859-1")


our_data["relevance"].value_counts()/our_data.shape[0] #Class distribution in the dataset
# convert label to a numerical variable
our_data = our_data[our_data.relevance != "not sure"]
our_data['relevance'] = our_data.relevance.map({'yes':1, 'no':0}) #relevant is 1, not-relevant is 0. 
our_data = our_data[["text","relevance"]] #Let us take only the two columns we need.
print("Shape of the data:", our_data.shape)



#train-test split
X = our_data.text #the column text contains textual data to extract features from
y = our_data.relevance #this is the column we are learning to predict. 
print("Shape of x, y")
print(X.shape, y.shape)
# split X and y into training and testing sets. By default, it splits 75% training and 25% test
#random_state=1 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("Shape of train and test after splitting")
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Pre-processing and vectorizing
stopwords = stop_words
def clean(doc): #doc is a string of text
    doc = doc.replace("</br>", " ") #This text contains a lot of <br/> tags.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    #remove punctuation and numbers
    return doc

#Preprocess and Vectorize train and test data
vect = CountVectorizer(preprocessor=clean) #instantiate a vectoriezer
X_train_dtm = vect.fit_transform(X_train)#use it to extract features from training data
#transform testing data (using training data's features)
X_test_dtm = vect.transform(X_test)
print("Train and test after extracting features: the dimension of our feature vector is 49753!")
print(X_train_dtm.shape, X_test_dtm.shape)

#Train a classifier
vect = CountVectorizer(preprocessor=clean, max_features=1000) #Step-1
X_train_dtm = vect.fit_transform(X_train)#combined step 2 and 3
X_test_dtm = vect.transform(X_test)

classifier = LogisticRegression(class_weight='balanced') #instantiate a logistic regression model
classifier.fit(X_train_dtm, y_train) #fit the model with training data

#Make predictions on test data
y_pred_class = classifier.predict(X_test_dtm)

#calculate evaluation measures:
print("Accuracy: ", accuracy_score(y_test, y_pred_class))

#Until here, it is the same code as earlier. 



#Part 2: Using Lime to interpret predictions

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

y_pred_prob = classifier.predict_proba(X_test_dtm)[:, 1]
c = make_pipeline(vect, classifier)
mystring = list(X_test)[221] #Take a string from test instance
print("Probability for", mystring)
print(c.predict_proba([mystring])) #Prediction is a "No" here. i.e., not relevant
class_names = ["no", "yes"] #not relevant, relevant
explainer = LimeTextExplainer(class_names=class_names)
print("Explanation:")
exp = explainer.explain_instance(mystring, c.predict_proba, num_features=6)
print(exp.as_list())

fig = exp.as_pyplot_figure()