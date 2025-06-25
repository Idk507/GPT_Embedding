import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
import torch.optim as optim

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size,embed_dim):
        super(TokenEmbedding, self).__init__()
        self.voacab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        nn.init.normal_(self.embedding.weight,mean=0.0,std=0.02)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)
