#necessary imports
import os
import pandas as pd
import wget
import tarfile

data_path = 'data/Chapter04'
    
if not os.path.exists('data/Chapter04/dbpedia_csv') :
    # downloading the data
    url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz"
    wget.download(url,data_path)

    # untaring the required file
    temp = data_path+'/dbpedia_csv.tar.gz'
    tar = tarfile.open(temp, "r:gz")
    tar.extractall(data_path)     
    tar.close()

print("Loading train and test data")
# Loading train data
train_file = data_path + '/dbpedia_csv/train.csv'
df = pd.read_csv(train_file, header=None, names=['class','name','description'])
# Loading test data
test_file = data_path + '/dbpedia_csv/test.csv'
df_test = pd.read_csv(test_file, header=None, names=['class','name','description'])
# Data we have
print("Train:{} Test:{}".format(df.shape,df_test.shape))



# Since we have no clue about the classes lets build one
# Mapping from class number to class name
class_dict={
            1:'Company',
            2:'EducationalInstitution',
            3:'Artist',
            4:'Athlete',
            5:'OfficeHolder',
            6:'MeanOfTransportation',
            7:'Building',
            8:'NaturalPlace',
            9:'Village',
            10:'Animal',
            11:'Plant',
            12:'Album',
            13:'Film',
            14:'WrittenWork'
        }
print("Mapping the classes")
df['class_name'] = df['class'].map(class_dict)
df.head()

df["class_name"].value_counts()

print("Clean text: remove issues, lowercase, normalize")
# Lets do some cleaning of this text
def clean_it(text,normalize=True):
    # Replacing possible issues with data. We can add or reduce the replacemtent in this chain
    s = str(text).replace(',',' ').replace('"','').replace('\'',' \' ').replace('.',' . ').replace('(',' ( ').\
            replace(')',' ) ').replace('!',' ! ').replace('?',' ? ').replace(':',' ').replace(';',' ').lower()
    
    # normalizing / encoding the text
    if normalize:
        s = s.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8')
    
    return s

# Now lets define a small function where we can use above cleaning on datasets
def clean_df(data, cleanit= False, shuffleit=False, encodeit=False, label_prefix='__class__'):
    # Defining the new data
    df = data[['name','description']].copy(deep=True)
    df['class'] = label_prefix + data['class'].astype(str) + ' '
    
    # cleaning it
    if cleanit:
        df['name'] = df['name'].apply(lambda x: clean_it(x,encodeit))
        df['description'] = df['description'].apply(lambda x: clean_it(x,encodeit))
    
    # shuffling it
    if shuffleit:
        df.sample(frac=1).reset_index(drop=True)
            
    return df



# Transform the datasets using the above clean functions
df_train_cleaned = clean_df(df, True, True)
df_test_cleaned = clean_df(df_test, True, True)


print("Write files to disk")
# Write files to disk as fastText classifier API reads files from disk.
train_file = data_path + '/dbpedia_train.csv'
df_train_cleaned.to_csv(train_file, header=None, index=False, columns=['class','name','description'] )

test_file = data_path + '/dbpedia_test.csv'
df_test_cleaned.to_csv(test_file, header=None, index=False, columns=['class','name','description'] )

print("Using fastText for feature extraction and training")
from fasttext import train_supervised 
"""fastText expects and training file (csv), a model name as input arguments.
label_prefix refers to the prefix before label string in the dataset.
default is __label__. In our dataset, it is __class__. 
There are several other parameters which can be seen in: 
https://pypi.org/project/fasttext/
"""
model = train_supervised(input=train_file, label="__class__", lr=1.0, epoch=75, loss='ova', wordNgrams=2, dim=200, thread=2, verbose=100)

for k in range(1,6):
    results = model.test(test_file,k=k)
    print(f"Test Samples: {results[0]} Precision@{k} : {results[1]*100:2.4f} Recall@{k} : {results[2]*100:2.4f}")

prediction = model.predict("Feroze Khan (Urdu: فیروز خان; born 11 July 1990)[2] is a Pakistani actor, model and video jockey.[3] who works in Urdu television. He made his acting debut with Bikhra Mera Naseeb as Harib and later on played various roles. Khan got his breakthrough after portraying Adeel in Hum TV's romantic Gul-e-Rana in 2015. His other notable works including Khaani, Ishqiya and Khuda Aur Muhabbat 3.", k=3)
print("Prediction for Feroze Khan (actor): \n", prediction)
