from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from spacy_utils import lemmatize, tokenize

# Let's say we choose a max sequence length of 10 (or any value that suits your dataset)
MAX_SEQ_LENGTH = 10

def tokenize_and_pad(sentence, all_words, max_seq_length):
    # Tokenize and get word indices (you can use vocab indices if you have them)
    tokens = [lemmatize(word) for word in tokenize(sentence)]
    
    # Convert tokens to indices in `all_words` list (like a vocabulary)
    token_indices = [all_words.index(word) for word in tokens if word in all_words]
    
    # Pad the sentence if it's shorter than max_seq_length
    if len(token_indices) < max_seq_length:
        token_indices += [0] * (max_seq_length - len(token_indices))  # Padding with 0
    else:
        token_indices = token_indices[:max_seq_length]  # Truncate if too long
    
    return token_indices
