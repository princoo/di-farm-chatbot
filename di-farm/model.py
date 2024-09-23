import torch
from torch import nn


class RnnModelV1(nn.Module):
    def __init__(self,
                 n_embeddings: int,
                 n_embedding_dim:int,
                 padding_idx:int,
                 n_hidden_layer:int,
                 n_hidden_layer_neurons: int,
                 n_classes:int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=n_embedding_dim,
            padding_idx=padding_idx
        )
        self.lstm_layer = nn.LSTM(
            input_size=n_embedding_dim,
            hidden_size=n_hidden_layer_neurons,
            batch_first=True,
            dropout=0.10
        )
        self.dence_layer = nn.Sequential(
            nn.Linear(in_features=9216, out_features=1024),
            nn.Dropout(0.5,inplace=True),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=1024,
                                    out_features=n_classes)
    
    def forward(self,token_type_ids,tag=None):
        embeds = self.embedding_layer(token_type_ids)
        lstm_out, hidden = self.lstm_layer(embeds)
        reshaped_out = torch.clone(lstm_out.reshape(-1,lstm_out.size(1) * lstm_out.size(2)))
        out = self.dence_layer(reshaped_out)
        out = self.classifier(torch.clone(out))
        return out
