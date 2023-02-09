#from https://amarsaini.github.io/Epoching-Blog/jupyter/nlp/pytorch/fastai/huggingface/2021/06/27/NLP-from-Scratch-with-PyTorch-FastAI-and-HuggingFace.html

import os
from zipfile import ZipFile
import tarfile
from fastai.text.all import *
import pandas as pd

data_path = os.getcwd() + "/data/Chapter04"
reviews_zip_path = data_path + '/archive.zip'
reviews_tgz_path = data_path + "/amazon_review_polarity_csv.tgz"
amazon_reviews_path = data_path + "/amazon_review_polarity_csv"

if not os.path.exists(amazon_reviews_path):
    with ZipFile(reviews_zip_path) as file:          
        file.extractall(data_path)      
    with tarfile.open(reviews_tgz_path) as file:
        file.extractall(data_path)    

train_df = pd.read_csv(amazon_reviews_path + '/train.csv', names=['label', 'title', 'text'], nrows=40000)
valid_df = pd.read_csv(amazon_reviews_path + '/test.csv', names=['label', 'title', 'text'], nrows=2000)
print(train_df.head())

sample_text = train_df['text'][0]
print(sample_text)

import torch
import torchtext
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")

sample_tokens = L(tokenizer(sample_text))
print("Tokens for sample text:")
print(sample_tokens)

from collections import Counter

token_counter = Counter()

for text in train_df['text']:
    tokens = tokenizer(text)
    token_counter.update(tokens)

print("25 most common tokens")
print(token_counter.most_common(n=25))

print("25 least common tokens")
print(token_counter.most_common()[-25:])

print("Token counter for \"well@@\" and \"well\", not recognized as the same word:")
print(token_counter['well@@'], token_counter['well'])

print("Building vocabulary...")

sorted_counter = dict(token_counter.most_common())

# Create vocab containing tokens with a minimum frequency of 20
my_vocab = torchtext.vocab.vocab(sorted_counter, min_freq=20)

# Add the unknown token, and use this by default for unknown words
unk_token = '<unk>'
my_vocab.insert_token(unk_token, 0)
my_vocab.set_default_index(0)

# Add the pad token
pad_token = '<pad>'
my_vocab.insert_token(pad_token, 1)

print("Vocab size and 25 first tokens")
print(len(my_vocab.get_itos()), my_vocab.get_itos()[:25])

glove = torchtext.vocab.GloVe(name = '6B', dim = 100)
print("Shape of GloVe vectors")
print(glove.vectors.shape)

print("Shape of my_vocab vectors after transferring from Glove, those than don't exist are initialized with a vector of 0s")
my_vocab.vectors = glove.get_vecs_by_tokens(my_vocab.get_itos())
print(my_vocab.vectors.shape)

tot_transferred = 0
for v in my_vocab.vectors:
    if not v.equal(torch.zeros(100)):
        tot_transferred += 1
        
tot_transferred, len(my_vocab)

tot_transferred = 0
for v in my_vocab.vectors:
    if not v.equal(torch.zeros(100)):
        tot_transferred += 1
        
print("Total of vocabulary transferred from GloVe")
print(tot_transferred, "of", len(my_vocab))

print("Vector for 'the'")
print(my_vocab.get_itos()[3], "-->", my_vocab.vectors[3])

print("Vector for 'eargels', out of vocabulary")
print(my_vocab.get_itos()[6555], "-->",  my_vocab.vectors[6555])

#If vector for out of vocabulary is initialized with 0s it may hinder the training of the model
print("Initialize vectors for out of vocabulary with random instead of 0s")

for i in range(my_vocab.vectors.shape[0]):
    if my_vocab.vectors[i].equal(torch.zeros(100)):
        my_vocab.vectors[i] = torch.randn(100)

print("Vector for 'eargels', out of vocabulary after adding random values")
print(my_vocab.get_itos()[6555], "-->",  my_vocab.vectors[6555])

print("Numericalize the sample tokens...")

numericalized_tokens = [my_vocab[token] for token in sample_tokens]
numericalized_tokens = torch.tensor(numericalized_tokens)
print(numericalized_tokens)

print("Sample tokens in Glove Vocab")
print(' '.join([my_vocab.get_itos()[num] for num in numericalized_tokens]))

print("Give max number of tokens and add paddings:")

max_tokens = 128
numericalized_tokens = [my_vocab[token] for token in sample_tokens]

if len(numericalized_tokens) < max_tokens:
    numericalized_tokens += [1] * (max_tokens-len(numericalized_tokens))
else:
    numericalized_tokens = numericalized_tokens[:max_tokens]

numericalized_tokens = torch.tensor(numericalized_tokens)
print(numericalized_tokens)

### Dataset & DataLoaders [PyTorch & fastai]
from torch import nn

class Simple_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_tokens):
        self.df = df
        self.vocab = vocab
        self.max_length = max_tokens
        self.tokenizer = get_tokenizer("basic_english")
        
        # label 1 is negative sentiment and label 2 is positive sentiment
        self.label_map = {1:0, 2:1}
        
    def __len__(self):
        return len(self.df)

    def decode(self, numericalized_tokens):
        return ' '.join([self.vocab.get_itos()[num] for num in numericalized_tokens])

    def __getitem__(self, index):
        label, title, text = self.df.iloc[index]
        label = self.label_map[label]
        label = torch.tensor(label)

        tokens = tokenizer(text)
        numericalized_tokens = [my_vocab[token] for token in tokens]

        if len(numericalized_tokens) < max_tokens:
            numericalized_tokens += [1] * (max_tokens-len(numericalized_tokens))
        else:
            numericalized_tokens = numericalized_tokens[:max_tokens]

        numericalized_tokens = torch.tensor(numericalized_tokens)
        
        return numericalized_tokens, label

print("Create datasets...")
train_dataset = Simple_Dataset(train_df, vocab=my_vocab, max_tokens=128)
valid_dataset = Simple_Dataset(valid_df, vocab=my_vocab, max_tokens=128)

train_dl = DataLoader(train_dataset, bs=32, shuffle=True)
valid_dl = DataLoader(valid_dataset, bs=32)

print("Create dataloaders from datasets")
dls = DataLoaders(train_dl, valid_dl)

### 4. Model [PyTorch]
class Model(nn.Module):
  
    def __init__(self, vocab, num_classes):
        super().__init__()
        
        vocab_size, emb_size = vocab.vectors.shape
        self.emb = nn.Embedding(vocab_size, emb_size, _weight=vocab.vectors)
        
        self.lstm = nn.LSTM(input_size = emb_size, hidden_size = 64, batch_first = True, num_layers = 2)
        
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        
    def forward(self, batch_data):

        token_embs = self.emb(batch_data)
        
        outputs, (h_n, c_n) = self.lstm(token_embs)
        
        # Assuming a batch size of 32, h_n will have a shape of:
        
        # shape = 2, 32, 64
        last_hidden_state = h_n
        # shape = 32, 2, 64
        last_hidden_state = last_hidden_state.permute(1,0,2)
        # shape = 32, 128
        last_hidden_state = last_hidden_state.flatten(start_dim=1)

        logits = self.head(last_hidden_state)
        
        return logits

model = Model(my_vocab, num_classes=2)

print("Let's check the model:")
print(model)

embedding_matrix = list(model.emb.parameters())[0]
print("Let's double check that some of our embeddings were successfully loaded from the domain-overlapping tokens from GloVe")
# print(embedding_matrix)
print("Index 3 corresponds to 'the'?")
print(my_vocab.vectors[3].equal(embedding_matrix[3]))

total_params = 0
for p in model.parameters():
    total_params += p.numel()
print("Number of parameters: ", total_params)

print("Now let's go ahead and make sure we can do a forward pass through our model, our loss function will be CrossEntropyLoss as it's a classification task.")
batched_data, batched_labels = train_dl.one_batch()
print(batched_data.shape, batched_labels.shape)

with torch.no_grad():
    logits = model(batched_data)
logits.shape

loss_func = nn.CrossEntropyLoss()

loss = loss_func(logits, batched_labels)
print("Loss:", loss)

### 5. Training/Fitting [fastai]
##para mañana