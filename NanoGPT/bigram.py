import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32  # embedding dimension
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
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
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

        X = token_embeddings + position_embeddings  # (B, T, C)
        logits = self.lm_head(X)  # (B, T, vocab_size)

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


model = BigramLanguageModel()
model = model.to(device)

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
