# The code is to be deployed on Lambda.ai
"""
    Train performance:
    step 0: train loss 4.2849, val loss 4.2823
    step 100: train loss 2.4721, val loss 2.4879
    step 200: train loss 2.4071, val loss 2.4355
    step 300: train loss 2.2862, val loss 2.3170
    step 400: train loss 2.0909, val loss 2.1511
    step 500: train loss 1.9357, val loss 2.0434
    step 600: train loss 1.8119, val loss 1.9513
    step 700: train loss 1.7299, val loss 1.8891
    step 800: train loss 1.6488, val loss 1.8184
    step 900: train loss 1.5954, val loss 1.7776
    step 1000: train loss 1.5404, val loss 1.7308
    step 1100: train loss 1.5012, val loss 1.6976
    step 1200: train loss 1.4673, val loss 1.6730
    step 1300: train loss 1.4342, val loss 1.6460
    step 1400: train loss 1.4091, val loss 1.6230
    step 1500: train loss 1.3841, val loss 1.6180
    step 1600: train loss 1.3640, val loss 1.6042
    step 1700: train loss 1.3450, val loss 1.5861
    step 1800: train loss 1.3254, val loss 1.5744
    step 1900: train loss 1.3061, val loss 1.5617
    step 2000: train loss 1.2949, val loss 1.5566
    step 2100: train loss 1.2770, val loss 1.5491
    step 2200: train loss 1.2689, val loss 1.5491
    step 2300: train loss 1.2538, val loss 1.5392
    step 2400: train loss 1.2397, val loss 1.5334
    step 2500: train loss 1.2268, val loss 1.5378
    step 2600: train loss 1.2127, val loss 1.5319
    step 2700: train loss 1.2018, val loss 1.5376
    step 2800: train loss 1.1909, val loss 1.5335
    step 2900: train loss 1.1808, val loss 1.5356
    step 3000: train loss 1.1641, val loss 1.5288
    step 3100: train loss 1.1512, val loss 1.5358
    step 3200: train loss 1.1424, val loss 1.5457
    step 3300: train loss 1.1335, val loss 1.5469
    step 3400: train loss 1.1164, val loss 1.5457
    step 3500: train loss 1.1058, val loss 1.5517
    step 3600: train loss 1.0948, val loss 1.5589
    step 3700: train loss 1.0822, val loss 1.5623
    step 3800: train loss 1.0699, val loss 1.5594
    step 3900: train loss 1.0604, val loss 1.5719
    step 4000: train loss 1.0467, val loss 1.5687
    step 4100: train loss 1.0289, val loss 1.5939
    step 4200: train loss 1.0172, val loss 1.6043
    step 4300: train loss 1.0011, val loss 1.6035
    step 4400: train loss 0.9887, val loss 1.6144
    step 4500: train loss 0.9784, val loss 1.6201
    step 4600: train loss 0.9626, val loss 1.6369
    step 4700: train loss 0.9512, val loss 1.6501
    step 4800: train loss 0.9337, val loss 1.6583
    step 4900: train loss 0.9213, val loss 1.6652
    step 4999: train loss 0.9069, val loss 1.6828
"""

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 1
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "mps"
eval_iters = 200
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Loaded text")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# create a mapping from characters to integers and vice versa
def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embdding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embdding_table(idx)  # (B,T,C)
        pos_emb = self.position_embdding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
