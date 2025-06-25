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
from  GPTEmbeddingConfig import  GPTEmbeddingConfig

config = GPTEmbeddingConfig(
    vocab_size=50257,  # GPT-2 vocabulary size
    embed_dim=768,     # GPT-2 small embedding dimension
    max_seq_len=1024,  # Maximum sequence length
    dropout=0.1,       # Dropout rate
    use_sinusoidal=False  # Use learned positional embeddings
)

# Create embedding layer
embedding_layer = create_gpt_embedding(config)

batch_size = 2
seq_len = 10

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print(f"Input shape: {input_ids.shape}")
print(f"Input IDs: {input_ids}")

embeddings = embedding_layer(input_ids)

embeddings
print(f"Output embeddings shape: {embeddings.shape}")
print(f"Expected shape: ({batch_size}, {seq_len}, {config.embed_dim})")

embeddings_with_ln = embedding_layer(input_ids, use_layer_norm=True)
print(f"Output with layer norm shape: {embeddings_with_ln.shape}")

token_emb = embedding_layer.token_embedding(input_ids)
print(f"Token embeddings shape: {token_emb.shape}")

pos_emb = embedding_layer.pos_embedding(token_emb)
print(f"Positional embeddings shape: {pos_emb.shape}")

config_sin = GPTEmbeddingConfig(use_sinusoidal=True)
embedding_layer_sin = create_gpt_embedding(config_sin)
embeddings_sin = embedding_layer_sin(input_ids)
print(f"Sinusoidal embeddings shape: {embeddings_sin.shape}")

total_params = sum(p.numel() for p in embedding_layer.parameters())
print(f"\nTotal parameters in embedding layer: {total_params:,}")

token_params = embedding_layer.token_embedding.embedding.weight.numel()

pos_params = embedding_layer.pos_embedding.positional_embedding.weight.numel()
pos_params

ln_params = sum(p.numel() for p in embedding_layer.layer_norm.parameters())
ln_params

print(f"Token embedding parameters: {token_params:,}")
print(f"Positional embedding parameters: {pos_params:,}")
print(f"Layer norm parameters: {ln_params:,}")

#testing gradient flow 
embeddings.sum().backward()

print(f"\nEmbedding statistics:")
print(f"Mean: {embeddings.mean().item():.4f}")
print(f"Std: {embeddings.std().item():.4f}")
print(f"Min: {embeddings.min().item():.4f}")
print(f"Max: {embeddings.max().item():.4f}")

