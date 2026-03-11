import os
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}")


# Get all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Create mappings between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# Encode: string → list of integers
encode = lambda s: [stoi[c] for c in s]
# Decode: list of integers → string
decode = lambda l: ''.join([itos[i] for i in l])

# Test it
print(encode("hello"))        # [46, 43, 50, 50, 53]
print(decode(encode("hello"))) # "hello"

# Encode the entire text
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}")  # [1115394] -- a 1D tensor of integers

# Split into train (90%) and validation (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

# Hyperparameters
batch_size = 64      # how many sequences to process in parallel
block_size = 256     # maximum context length
n_embd = 384         # embedding dimension
n_head = 6           # number of attention heads
n_layer = 6          # number of transformer blocks
dropout = 0.2        # regularization (explained below)
learning_rate = 3e-4 # how big of a step to take each update
max_iters = 5000     # total training steps
eval_interval = 500  # how often to check validation loss
eval_iters = 200     # how many batches to average for validation loss estimate


def get_batch(split):
    """Get a random batch of training data."""
    data_split = train_data if split == 'train' else val_data
    # Pick random starting positions
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    # x is the input, y is the target (shifted by one position)
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Test it
xb, yb = get_batch('train')
print(f"Input shape:  {xb.shape}")   # [64, 256]
print(f"Target shape: {yb.shape}")   # [64, 256]

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask: prevent attending to future positions
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Compute attention scores ("@" is matrix multiplication, same as torch.matmul)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        # Mask out future positions (causal attention)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """The feed-forward network: two linear layers with GELU activation."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand to 4x width
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),  # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention + feed-forward, with residuals and layer norm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # attention with residual connection
        x = x + self.ffwd(self.ln2(x))  # feed-forward with residual connection
        return x


class GPT(nn.Module):
    """The full GPT model."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token embeddings + position embeddings
        tok_emb = self.token_embedding_table(idx)           # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                               # (B, T, n_embd)

        # Pass through all transformer blocks
        x = self.blocks(x)                                   # (B, T, n_embd)
        x = self.ln_f(x)                                     # (B, T, n_embd)
        logits = self.lm_head(x)                             # (B, T, vocab_size)

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens using autoregressive generation.

        "Autoregressive" means: generate one token, append it to the input,
        then use the extended input to generate the next token. Each token
        is conditioned on all previous tokens (including previously generated
        ones). This is how every GPT-style model generates text -- one token
        at a time, feeding its own output back as input.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (can't exceed our position embeddings)
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append to the sequence and use it as input for the next step
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Create the model
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters ({n_params/1e6:.1f}M)")

# Generate from the untrained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with a newline
print("=== UNTRAINED MODEL OUTPUT ===")
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
print("==============================")

@torch.no_grad()
def estimate_loss():
    """Average loss over multiple batches for a more stable estimate."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for iter in range(max_iters):

    # Every eval_interval steps, evaluate on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:>5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)

    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Optimizer step: update weights
    optimizer.step()

print("Training complete!")

# Generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("=== TRAINED MODEL OUTPUT ===")
generated = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated)
print("============================")


def generate_from_prompt(prompt, max_new_tokens=300):
    """Generate text starting from a given prompt."""
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(output[0].tolist())

# Try some prompts
prompts = [
    "ROMEO:",
    "To be or not to be",
    "The king",
    "JULIET:\nO Romeo, Romeo,",
    "First Citizen:\nWe are",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    print(generate_from_prompt(prompt, max_new_tokens=200))
    
# Save the model weights
torch.save({
    'model_state_dict': model.state_dict(),
    'chars': chars,
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
}, 'shakespeare_gpt.pt')

print(f"Model saved! File size: {os.path.getsize('shakespeare_gpt.pt') / 1e6:.1f} MB")