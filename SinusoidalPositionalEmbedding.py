import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
import torch.optim as optim



class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len,embed_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0,max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,embed_dim,2).float()* - (math.log(10000.0)/embed_dim))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self,x):
        batch_size,seq_len =x.shape[0],x.shape[1]
        return self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
    
