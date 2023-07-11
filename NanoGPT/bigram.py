import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 16  # 64  # how many independent sequences will we process in parallel?
block_size = 32  # 256  # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 1e-3  # 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 64  # 384  # embedding dimension
n_attn_heads = 4
n_layers = 4
dropout = 0.2
# ---

# Dataset loading/utils
torch.manual_seed(1337)
# read it in to inspect it
with open("../tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train/test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # 4 numbers randomly generated between 0 and len(data) -  block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ---


@torch.no_grad()
def estimate_loss(model):
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


# Simplest model version
class SimpleBigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers where B = Batch size (4) and T = Time/Context Size (8)
        token_embeddings = self.token_embedding_table(
            idx
        )  # (B, T, C) where C is channel/embedding dimension (vocab size, 65)

        logits = self.lm_head(token_embeddings)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape the logits tensor so that it is (B*T, C) because PyTorch expects the channel dimension to be the second dimension
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate (B, T + 1), (B, T + 2), ... (B, T + max_new_tokens)
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


# ---


# Writing up self-attention!
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # A non-parameter of the model is called a "buffer" in PyTorch
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # C = head_size
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores (affinities)
        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Compute the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = weights @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # concatenate over the channel dimension
        out = self.proj(
            out
        )  # Just a linear transformation over the outcome of the multihead attention layer
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embed, n_embed * 4
            ),  # The multiplication by 4 is taken from the paper
            nn.ReLU(),
            nn.Linear(
                n_embed * 4, n_embed
            ),  # this is the projection layer, same as multihead attention!
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A Transformer block: Communication (Multihead self-attention) followed by Computation (Feed-forward)"""

    def __init__(self, n_embed, n_attn_heads):
        super().__init__()
        head_size = n_embed // n_attn_heads  # 32 // 4 = 8
        self.sa = MultiHeadAttention(n_attn_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        # Per token transformation/normalization of each token's features
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # x = self.sa(x)
        # x = self.ffwd(x)
        # Adding residual connections: An uninterrupted pathway from the prediction to the input.
        # Since during backpropagation, the addition operation distributes gradients equally to both of its input branches,
        #   this allows for the gradients from the loss to hop all the way to the input, AND fork off the residual blocks (the other operations)
        #   in fact, these other "residual" blocks/pathways contribute very little at the beginning/initialization because they're usually initialized that way, and over time they start to contribute more during the optimization process.
        #   this is a good optimization!!!!!
        x = x + self.sa(self.ln1(x))  # Fork off, do some communication, then come back
        x = x + self.ffwd(self.ln2(x))  # Fork off, do some computation, then come back
        return x


# Language model with self-attention!
class LanguageModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # self.sa_head = SelfAttentionHead(n_embed)
        # self.sa_heads = MultiHeadAttention(
        #     n_attn_heads, n_embed // n_attn_heads
        # )  # i.e 4 heads of 8-dimensional self-attention = 4 communication channels in parallel, when concatenated it gives the original n_embed
        # self.ffwd = FeedForward(n_embed)

        self.transformer_blocks = nn.Sequential(
            *[Block(n_embed, n_attn_heads) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(
            n_embed
        )  # Final LayerNorm after Transformer blocks
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers where B = Batch size (4) and T = Time/Context Size (8)
        token_embeddings = self.token_embedding_table(
            idx
        )  # (B, T, C) where C is channel/embedding dimension (vocab size, 65)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)

        x = token_embeddings + position_embeddings  # (B, T, C)

        # X = self.sa_head(X)  # Apply one head of self-attention: (B, T, C)
        # x = self.sa_heads(x)  # Apply multiple heads of self-attention: (B, T, C)
        # Once all that information has been aggregated, do a linear + non-linear operation to "think" about that information individually, at the token level
        # x = self.ffwd(x)  # (B, T, C)

        x = self.transformer_blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape the logits tensor so that it is (B*T, C) because PyTorch expects the channel dimension to be the second dimension
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate (B, T + 1), (B, T + 2), ... (B, T + max_new_tokens)
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context that we'll feed to self to avoid running into index errors when looking at positional encodings
            # so that we never pass more than block_size elements
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


model = LanguageModelWithAttention()
model = model.to(device)

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("=" * 32)
print("Training...")
for iter in range(max_iters):
    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"Step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
print("=" * 32)
print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
