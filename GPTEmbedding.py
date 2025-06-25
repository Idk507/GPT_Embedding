import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
import torch.optim as optim
from TokenEmbedding import TokenEmbedding
from PositionalEmbedding import PositionalEmbedding
from SinusoidalPositionalEmbedding import SinusoidalPositionalEmbedding


class GPTEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_seq_len,dropout=0.1,use_sinusoidal= False):
        super(GPTEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        if use_sinusoidal:
            self.pos_embedding = SinusoidalPositionalEmbedding(max_seq_len, embed_dim)
        else:
            self.pos_embedding = PositionalEmbedding(max_seq_len, embed_dim)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #layer norm 
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self,input_ids,use_layer_norm = True):
        #token embedding
        token_embedding = self.token_embedding(input_ids)
        #positional embedding
        pos_embedding = self.pos_embedding(token_embedding)
        #add two embedding together
        embedding = token_embedding + pos_embedding
        #apply layer normalization 
        if use_layer_norm:
            embedding = self.layer_norm(embedding)
        #apply dropout
        embedding = self.dropout(embedding)
        return embedding

