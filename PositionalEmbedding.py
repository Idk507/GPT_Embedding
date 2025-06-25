import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
import torch.optim as optim

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len,embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        nn.init.normal_(self.positional_embedding.weight,mean=0.0,std=0.02)

    def forward(self, x):
        batch_size,seq_len = x.shape[0],x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.positional_embedding(positions) 
        
