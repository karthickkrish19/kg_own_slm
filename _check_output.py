#!/usr/bin/env python3
"""Check current best_model.pt output quality."""
import torch
from model import SLM
from tokenizer_utils import SLMTokenizer

tok = SLMTokenizer('out/tokenizer.json')
model = SLM.load('out/checkpoints/best_model.pt', map_location='cpu')
model.eval()

print(f"Model: {model.num_parameters():,} params | vocab={model.config.vocab_size}")
print(f"Tokenizer vocab: {tok.vocab_size}")

# Check if model vocab matches tokenizer vocab
if model.config.vocab_size != tok.vocab_size:
    print(f"\n*** MISMATCH: model vocab={model.config.vocab_size} vs tokenizer vocab={tok.vocab_size} ***")
    print("This is a critical problem!")
else:
    print("Model/tokenizer vocab match: OK")

print("\n=== GENERATION (temp=0.7, top_k=30) ===")
prompts = [
    "Once upon a time",
    "The art of war is",
    "Alice was beginning to get",
    "It was the best of times",
    "A Fox one day fell into",
    "There was once a king who had",
    "The general who wins a battle",
    "Down the Rabbit-Hole",
]

for p in prompts:
    ids = tok.encode(p, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=60, temperature=0.7,
                             top_k=30, repetition_penalty=1.3)
    text = tok.decode(out[0].tolist())
    print(f"\nPROMPT: {p}")
    print(f"OUTPUT: {text[:200]}")

# Token prediction accuracy
print("\n\n=== PREDICTION ACCURACY ===")
import random, math
random.seed(42)
with open("data/train.txt", encoding="utf-8") as f:
    text = f.read()

total_rank = 0
total_tokens = 0
top1 = top5 = top10 = 0

for _ in range(20):
    start = random.randint(0, len(text) - 300)
    passage = text[start:start+200].strip()
    ids = tok.encode(passage, add_special_tokens=False)
    if len(ids) < 5:
        continue
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, _, _ = model(x)
    for i in range(len(ids) - 1):
        probs = torch.softmax(logits[0, i], dim=-1)
        sorted_ids = probs.argsort(descending=True)
        rank = (sorted_ids == ids[i+1]).nonzero(as_tuple=True)[0].item() + 1
        total_rank += rank
        total_tokens += 1
        if rank == 1: top1 += 1
        if rank <= 5: top5 += 1
        if rank <= 10: top10 += 1

print(f"Tokens tested: {total_tokens}")
print(f"Top-1:  {top1/total_tokens*100:.1f}%")
print(f"Top-5:  {top5/total_tokens*100:.1f}%")
print(f"Top-10: {top10/total_tokens*100:.1f}%")
print(f"Avg rank: {total_rank/total_tokens:.1f}")

# Val perplexity
print("\n=== VAL PERPLEXITY ===")
from dataset import get_dataloaders
_, val_dl, _ = get_dataloaders(
    train_file="data/train.txt", tokenizer=tok,
    seq_len=256, batch_size=8, stride=128,
    val_file="data/val.txt", seed=42, pad_id=tok.pad_id,
    cache_dir="out/tokenized_cache",
)
total_loss = count = 0
with torch.no_grad():
    for i, (bx, by, bm) in enumerate(val_dl):
        _, loss, _ = model(bx, targets=by, loss_mask=bm)
        if loss is not None:
            total_loss += loss.item()
            count += 1
avg_val = total_loss / max(count, 1)
print(f"Val loss: {avg_val:.4f}  ppl={math.exp(min(avg_val, 88)):.1f}")
