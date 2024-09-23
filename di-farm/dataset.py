import torch
import numpy as np
from tokenizer import Tokenizer

class IntentDataset:
    def __init__(self,intents:np.ndarray,tags:np.ndarray,tokenizer:Tokenizer):
        self.intents = intents
        self.tags = tags
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.tags)
    
    def __getitem__(self,idx:int):
        tokenized_text_dict = self.tokenizer.tokenize(self.intents[idx])

        return{
            "token_type_ids": torch.tensor(
                tokenized_text_dict["token_type_ids"],dtype=torch.long
            ),
            "tag":torch.tensor(self.tags[idx], dtype=torch.long)
        }
