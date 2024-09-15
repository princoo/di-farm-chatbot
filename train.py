import json
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet
from sklearn.model_selection import train_test_split
from spacy_utils import tokenize,lemmatize,bag_of_words
from training_loop import train_step
from testing_loop import test_step

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
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
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
HIDDEN_UNITS = 8
OUTPUT_SIZE = len(tags)
INPUT_SIZE = len(all_words) # this would be the same also as the len of a single element of X_train
LEARNING_RATE = 0.001
EPOCHS = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = ChatDataset()
train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True,random_state=42)

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                        #   num_workers=2
                          )
test_loader = DataLoader(dataset=test_data,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                        #   num_workers=2
                          )
print(len(train_loader))
print(len(test_loader))
print(len(train_data))
print(len(test_data))
#  model 
modelV1 = NeuralNet(input_size=INPUT_SIZE,
                    hidden_units=HIDDEN_UNITS,
                    output_size=OUTPUT_SIZE).to(device)

#  setting up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelV1.parameters(), lr=LEARNING_RATE)

# for epoch in tqdm(range(EPOCHS)):
#     train_loss,train_acc = train_step(model=modelV1,
#                dataloader=train_loader,
#                loss_fn=loss_fn,
#                optimizer=optimizer,
#                device=device
#                )
#     test_loss,test_acc = test_step(model=modelV1,
#                dataloader=test_loader,
#                loss_fn=loss_fn,
#                device=device
#                )

#     if epoch % 100 ==0:
#         print (f"\n Epoch:{epoch} -------- \n")
#         print(f"Train loss:{train_loss:.5f} | Train acc:{train_acc:.2f}%")
#         print(f"Train loss:{test_loss:.5f} | Train acc:{test_acc:.2f}%")