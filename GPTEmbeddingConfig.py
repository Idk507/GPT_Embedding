import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
import torch.optim as optim
from TokenEmbedding import TokenEmbedding
from PositionalEmbedding import PositionalEmbedding
from SinusoidalPositionalEmbedding import SinusoidalPositionalEmbedding
from GPTEmbedding import GPTEmbedding

class GPTEmbeddingConfig:
    def __init__(self, vocab_size=50257, embed_dim=768, max_seq_len=1024, 
                 dropout=0.1, use_sinusoidal=False):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_sinusoidal = use_sinusoidal

def create_gpt_embedding(config):
    return GPTEmbedding(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_sinusoidal=config.use_sinusoidal
    )

