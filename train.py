import json
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
# from model import NeuralNet
from model import Seq2Seq, EncoderLSTM, DecoderLSTM
from spacy_utils import tokenize,lemmatize,bag_of_words
from training_loop import train_step
from save_model import save_model_version
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
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Convert X_train to tensor
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)     # Convert y_train to tensor
# Define a function to pad sequences
def pad_sequences(sequences, max_len):
    padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded_sequences[i, :length] = torch.tensor(seq[:length])
    return padded_sequences
class ChatDataset(Dataset):
    def __init__(self,X_data,y_data):
        self.x_data= X_data
        self.y_data = y_data
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)


# Hyper parameter
BATCH_SIZE = 8
HIDDEN_UNITS = 128
OUTPUT_SIZE = len(tags)
INPUT_SIZE = len(all_words) # this would be the same also as the len of a single element of X_train
LEARNING_RATE = 0.001
MANUAL_SEED = 42
EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          )

torch.manual_seed(MANUAL_SEED)
embed_size = INPUT_SIZE
hidden_size = 512
modelV1 = Seq2Seq(encoder=EncoderLSTM(embed_size, hidden_size),
                decoder=DecoderLSTM(hidden_size, OUTPUT_SIZE)).to(device)

#  setting up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(modelV1.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(EPOCHS)):
    train_loss,train_acc = train_step(model=modelV1,
               dataloader=train_loader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               device=device
               )

    if epoch % 10 ==0:
        print (f"\n Epoch:{epoch} -------- \n")
        print(f"Train loss:{train_loss:.5f} | Train acc:{train_acc:.2f}%")

# save_model_version(model=modelV1,
#                    input_size=INPUT_SIZE,
#                    hidden_units=HIDDEN_UNITS,
#                    output_size=OUTPUT_SIZE,
#                    all_words=all_words,
#                    tags=tags)