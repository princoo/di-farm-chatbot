import json
from spacy_utils import tokenize,lemmatize,bag_of_words
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

with open("intents.json","r") as f:
    intents = json.load(f)

all_words=[]
tags = []
xy=[]

for intent in intents["intents"]:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?','!',',','.']
all_words=[lemmatize(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # this will remove all duplicate words
tags = sorted(set(tags)) # this will remove all duplicate tags

X_train=[]
y_train=[]

for pattern_sentence,tag in xy:
    bag=bag_of_words(pattern_sentence,tag)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train,y_train = np.array(X_train), np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data= X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hyper parameter
BATCH_SIZE = 8 
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)

