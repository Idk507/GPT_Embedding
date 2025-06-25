# GPT Embedding Implementation from Scratch

A complete, production-ready implementation of GPT-style embeddings in PyTorch, built from scratch without external dependencies beyond PyTorch.

## 🚀 Features

- **Complete GPT Embedding Layer**: Token + Positional embeddings with proper initialization
- **Multiple Positional Encoding Options**: Both learned and sinusoidal positional embeddings
- **Production Ready**: Includes proper error handling, device management, and gradient flow
- **Configurable**: Easy-to-use configuration system for different model sizes
- **Well Documented**: Comprehensive code documentation and examples
- **Testing Included**: Built-in testing and validation functions

## 📋 Requirements

```bash
torch>=1.9.0
numpy>=1.19.0
```

## 🏗️ Architecture Overview

The GPT embedding layer consists of three main components:

```
Input Token IDs → Token Embedding → 
                                  ↘
                                   Add → Layer Norm (optional) → Dropout → Output
                                  ↗
Position Indices → Positional Embedding →
```

### Components

1. **Token Embedding**: Converts token IDs to dense vectors
2. **Positional Embedding**: Encodes sequence position information
3. **Combination Layer**: Adds token and positional embeddings
4. **Regularization**: Optional layer normalization and dropout

## 🚀 Quick Start

### Basic Usage

```python
import torch
from gpt_embedding import GPTEmbedding, GPTEmbeddingConfig

# Create configuration
config = GPTEmbeddingConfig(
    vocab_size=50257,    # GPT-2 vocabulary size
    embed_dim=768,       # GPT-2 small embedding dimension
    max_seq_len=1024,    # Maximum sequence length
    dropout=0.1          # Dropout rate
)

# Initialize embedding layer
embedding_layer = GPTEmbedding(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout
)

# Prepare input
batch_size, seq_len = 2, 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
embeddings = embedding_layer(input_ids)
print(f"Output shape: {embeddings.shape}")  # [2, 10, 768]
```

### Advanced Configuration

```python
# GPT-2 Medium configuration
config_medium = GPTEmbeddingConfig(
    vocab_size=50257,
    embed_dim=1024,      # Larger embedding dimension
    max_seq_len=1024,
    dropout=0.1,
    use_sinusoidal=False # Use learned positional embeddings
)

# GPT-2 Large configuration
config_large = GPTEmbeddingConfig(
    vocab_size=50257,
    embed_dim=1280,
    max_seq_len=1024,
    dropout=0.1
)

# Custom configuration with sinusoidal embeddings
config_custom = GPTEmbeddingConfig(
    vocab_size=32000,    # Custom vocabulary
    embed_dim=512,       # Smaller model
    max_seq_len=2048,    # Longer sequences
    dropout=0.2,         # Higher dropout
    use_sinusoidal=True  # Fixed positional embeddings
)
```

## 📊 Model Configurations

| Model Size | Vocab Size | Embed Dim | Parameters (Embedding) |
|------------|------------|-----------|------------------------|
| GPT-2 Small| 50,257     | 768       | ~39.4M                |
| GPT-2 Medium| 50,257    | 1,024     | ~52.5M                |
| GPT-2 Large| 50,257     | 1,280     | ~65.1M                |
| GPT-2 XL   | 50,257     | 1,600     | ~81.2M                |

## 🔧 API Reference

### GPTEmbeddingConfig

Configuration class for GPT embeddings.

```python
class GPTEmbeddingConfig:
    def __init__(self, 
                 vocab_size=50257,      # Vocabulary size
                 embed_dim=768,         # Embedding dimension
                 max_seq_len=1024,      # Maximum sequence length
                 dropout=0.1,           # Dropout rate
                 use_sinusoidal=False   # Use sinusoidal embeddings
                ):
```

### GPTEmbedding

Main embedding layer class.

```python
class GPTEmbedding(nn.Module):
    def forward(self, input_ids, use_layer_norm=False):
        """
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len)
            use_layer_norm (bool): Whether to apply layer normalization
            
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, seq_len, embed_dim)
        """
```

### Individual Components

#### TokenEmbedding
```python
token_embedding = TokenEmbedding(vocab_size=50257, embed_dim=768)
token_embeds = token_embedding(input_ids)
```

#### PositionalEmbedding
```python
# Learned positional embeddings
pos_embedding = PositionalEmbedding(max_seq_len=1024, embed_dim=768)

# Sinusoidal positional embeddings
sin_embedding = SinusoidalPositionalEmbedding(max_seq_len=1024, embed_dim=768)
```

## 🧪 Testing

The implementation includes comprehensive testing:

```python
# Run built-in tests
python gpt_embedding.py
```

### Test Coverage

- ✅ Forward pass functionality
- ✅ Gradient flow verification
- ✅ Shape consistency
- ✅ Device compatibility
- ✅ Parameter counting
- ✅ Statistical properties
- ✅ Both embedding types

## 🎯 Key Features Explained

### 1. Proper Initialization

```python
# Token embeddings initialized with small random values
nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
```

**Why std=0.02?**
- Prevents vanishing/exploding gradients
- Empirically proven to work well in large language models
- Maintains proper gradient flow during training

### 2. Embedding Scaling

```python
return self.embedding(x) * math.sqrt(self.embed_dim)
```

**Purpose:**
- Maintains consistent variance across different embedding dimensions
- Helps with gradient stability
- Balances token and positional embedding magnitudes

### 3. Flexible Positional Embeddings

**Learned Embeddings (Default):**
- Trainable parameters that adapt to your specific task
- Better performance on most downstream tasks
- Limited to training sequence length

**Sinusoidal Embeddings:**
- Fixed mathematical pattern, no additional parameters
- Can handle sequences longer than training data
- More memory efficient

### 4. Device Management

```python
positions = torch.arange(seq_len, device=x.device)
```

Ensures all tensors are on the same device (CPU/GPU) for compatibility.

## 📈 Performance Considerations

### Memory Usage

- **Token Embeddings**: `vocab_size × embed_dim × 4 bytes`
- **Positional Embeddings**: `max_seq_len × embed_dim × 4 bytes`
- **Activations**: `batch_size × seq_len × embed_dim × 4 bytes`

### Computational Complexity

- **Time Complexity**: O(seq_len) - linear in sequence length
- **Space Complexity**: O(batch_size × seq_len × embed_dim)

### Optimization Tips

1. **Use appropriate batch sizes** based on your GPU memory
2. **Choose embedding dimension** based on your task complexity
3. **Consider sinusoidal embeddings** for longer sequences
4. **Adjust dropout rate** based on your dataset size

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or sequence length
config = GPTEmbeddingConfig(max_seq_len=512)  # Instead of 1024
```

**2. Gradient Issues**
```python
# Check if gradients are flowing
embeddings.register_hook(lambda grad: print(f"Gradient norm: {grad.norm()}"))
```

**3. Device Mismatch**
```python
# Ensure model and data are on same device
model = model.to(device)
input_ids = input_ids.to(device)
```





