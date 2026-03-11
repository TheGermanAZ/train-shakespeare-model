# Train Shakespeare Model

A character-level GPT trained on Shakespeare's complete works. Generates new Shakespeare-like text after training on ~1MB of his plays.

Based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

## Architecture

| Component | Detail |
|-----------|--------|
| Model | GPT (decoder-only transformer) |
| Tokenizer | Character-level (65 unique characters) |
| Embedding dim | 384 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Context length | 256 |
| Parameters | ~10.8M |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TheGermanAZ/train-shakespeare-model.git
cd train-shakespeare-model
```

## Running with PyTorch

```bash
# Download the dataset
uv run python get_data.py

# Train the model and generate text
uv run python train.py
```

Uses MPS (Apple Silicon GPU) automatically when available, falls back to CUDA or CPU.

## Running with MLX (Apple Silicon)

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework optimized for Apple Silicon. It uses unified memory (no `.to(device)` calls) and lazy evaluation.

### Install MLX

```bash
uv add mlx
```

### Key differences from PyTorch

| PyTorch | MLX |
|---------|-----|
| `import torch` | `import mlx.core as mx` |
| `import torch.nn as nn` | `import mlx.nn as nn` |
| `torch.optim.AdamW(...)` | `mlx.optimizers.AdamW(...)` |
| `loss.backward()` | `nn.value_and_grad(model, loss_fn)` |
| `optimizer.step()` | `optimizer.update(model, grads)` |
| `x.to(device)` | Not needed (unified memory) |
| `model.eval()` / `model.train()` | `model.eval()` / `model.train()` |
| `torch.no_grad()` | Not needed (explicit grad via `value_and_grad`) |
| `F.cross_entropy(logits, targets)` | `nn.losses.cross_entropy(logits, targets)` |
| `def forward(self, x):` | `def __call__(self, x):` |

### MLX training loop pattern

The biggest structural change is how gradients work. PyTorch uses implicit autograd (`loss.backward()`), while MLX uses explicit function transformations:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

# This returns a function that computes both loss AND gradients
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.AdamW(learning_rate=3e-4)

for step in range(max_iters):
    x, y = get_batch('train')
    loss, grads = loss_and_grad_fn(model, x, y)

    # Update model params from gradients
    optimizer.update(model, grads)

    # MLX is lazy — force evaluation to actually compute
    mx.eval(model.parameters(), optimizer.state)
```

### MLX causal mask

PyTorch uses `register_buffer` + `masked_fill`. In MLX, use `mx.addmm` or manual masking:

```python
mask = mx.tril(mx.ones((T, T)))
weights = weights * mask + (1 - mask) * -1e9
weights = mx.softmax(weights, axis=-1)
```

### Generating text with MLX

```python
# MLX uses mx.random for sampling
probs = mx.softmax(logits[:, -1, :], axis=-1)
idx_next = mx.random.categorical(mx.log(probs), num_samples=1)
```

## Files

| File | Purpose |
|------|---------|
| `get_data.py` | Downloads the Tiny Shakespeare dataset |
| `train.py` | Model definition, training loop, text generation |
| `tokenizer.py` | (placeholder) Character-level tokenizer |
| `main.py` | Entry point |

## Sample output (after training)

```
ROMEO:
O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear...
```

(Actual output varies — the model generates novel Shakespeare-like text, not memorized quotes.)
