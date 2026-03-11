import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Download Shakespeare (~1MB of text) if not already present
if not os.path.exists('shakespeare.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'shakespeare.txt')
    print("Downloaded shakespeare.txt")
else:
    print("Using existing shakespeare.txt")

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}")