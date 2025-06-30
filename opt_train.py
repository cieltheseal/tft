import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import os

# ----------------------------
# Load CSV Data
# ----------------------------
def load_data_from_csv(path):
    df = pd.read_csv(path)
    # Ensure list format from stringified list
    df["input_units"] = df["input_units"].apply(ast.literal_eval)
    samples = [{"input_units": row["input_units"], "target_unit": row["label_unit"]} for _, row in df.iterrows()]
    return samples

# ----------------------------
# Build Vocabulary
# ----------------------------
def build_vocab(samples):
    all_units = set(u for s in samples for u in s["input_units"] + [s["target_unit"]])
    unit_to_idx = {unit: idx + 1 for idx, unit in enumerate(sorted(all_units))}  # +1 to reserve 0 for <PAD>
    unit_to_idx["<PAD>"] = 0
    idx_to_unit = {idx: unit for unit, idx in unit_to_idx.items()}
    return unit_to_idx, idx_to_unit

# ----------------------------
# Dataset
# ----------------------------
class BoardDataset(Dataset):
    def __init__(self, samples, unit_to_idx, pad_idx=0):
        self.samples = samples
        self.unit_to_idx = unit_to_idx
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        units = self.samples[idx]["input_units"]
        target = self.samples[idx]["target_unit"]
        input_ids = [self.unit_to_idx[u] for u in units]
        target_id = self.unit_to_idx[target]
        return {"input_ids": input_ids, "target_id": target_id}

# ----------------------------
# Collate Function
# ----------------------------
def collate_fn(batch, pad_idx=0):
    input_seqs = [item["input_ids"] for item in batch]
    target_ids = torch.tensor([item["target_id"] for item in batch])
    max_len = max(len(seq) for seq in input_seqs)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in input_seqs]
    input_ids = torch.tensor(padded)
    return {"input_ids": input_ids, "target_id": target_ids}

# ----------------------------
# Model
# ----------------------------
class BestUnitPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, input_ids):
        mask = (input_ids != self.pad_idx).float().unsqueeze(-1)
        embeds = self.embedding(input_ids)
        masked_embeds = embeds * mask
        sum_embeds = masked_embeds.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        mean_embeds = sum_embeds / counts
        return self.mlp(mean_embeds)

# ----------------------------
# Accuracy
# ----------------------------
def top_k_accuracy(logits, targets, k=1):
    topk = logits.topk(k, dim=1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

# ----------------------------
# Training / Validation
# ----------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_id"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, target_ids)
        acc = top_k_accuracy(logits, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_id"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, target_ids)
            acc = top_k_accuracy(logits, target_ids)

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

# ----------------------------
# Main Training Function
# ----------------------------
def run_training(train_csv_path, val_csv_path, epochs=10, batch_size=32):
    train_samples = load_data_from_csv(train_csv_path)
    val_samples = load_data_from_csv(val_csv_path)

    unit_to_idx, idx_to_unit = build_vocab(train_samples + val_samples)
    pad_idx = unit_to_idx["<PAD>"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BoardDataset(train_samples, unit_to_idx, pad_idx)
    val_ds = BoardDataset(val_samples, unit_to_idx, pad_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    model = BestUnitPredictor(vocab_size=len(unit_to_idx), pad_idx=pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_model(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, val_dl, criterion, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return model, unit_to_idx, idx_to_unit


# ----------------------------
# Run Training From CSV
# ----------------------------
if __name__ == "__main__":
    # Example usage — replace with your actual CSV paths
    train_csv = "data/train_opt.csv"
    val_csv = "data/val_opt.csv"

    assert os.path.exists(train_csv) and os.path.exists(val_csv), "CSV files not found."

    model, unit_to_idx, idx_to_unit = run_training(train_csv, val_csv, epochs=20)

    # Save model state and mappings
    torch.save({
        'model_state_dict': model.state_dict(),
        'unit_to_idx': unit_to_idx,
        'idx_to_unit': idx_to_unit
    }, 'optimiser.pt')