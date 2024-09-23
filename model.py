import torch
from torch import nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()  
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)
    def forward(self, input):
        outputs, (hidden, cell) = self.lstm(input)  
        # print(f"shape: {hidden.shape}")
        return outputs, hidden, cell
class DecoderLSTM(nn.Module): 
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, output_size,batch_first=True)
    def forward(self, input):
        outputs, _ = self.lstm(input)
        return outputs
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder): 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,input):
        # Pass the input through the encoder
        hidden, cell = self.encoder(input)
        # Pass the hidden state through the decoder
        outputs = self.decoder(hidden)
        return outputs

        return outputs
    

