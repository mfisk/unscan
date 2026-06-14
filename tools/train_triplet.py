#!/usr/bin/env python3
"""
Per-glyph triplet network trainer for unscan font classification.

Reads training data from gen_training_data output (features.bin + labels.jsonl),
trains one small embedding network per character, and exports a single binary
weights file consumable by unscan's TripletClassifier.

Architecture per glyph: 100 → 128 (ReLU) → 64 (ReLU) → 32 → L2-normalize
Loss: triplet margin loss with semi-hard negative mining

Usage:
    python3 tools/train_triplet.py training-data/ -o triplet-weights.bin
    python3 tools/train_triplet.py training-data/ -o triplet-weights.bin --epochs 50

Binary output format:
    Header:
        magic: b"TRIP" (4 bytes)
        version: u32 LE (1)
        n_chars: u32 LE
    Per character (repeated n_chars times):
        char_code: u32 LE (Unicode codepoint)
        W1: 100×128 f32 LE (row-major)
        b1: 128 f32 LE
        W2: 128×64 f32 LE
        b2: 64 f32 LE
        W3: 64×32 f32 LE
        b3: 32 f32 LE
    Total per char: 23,072 params × 4 bytes = 92,288 bytes
"""

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FEAT_DIM = 100
L1_OUT = 128
L2_OUT = 64
EMBED_DIM = 32
PARAMS_PER_MODEL = (FEAT_DIM * L1_OUT + L1_OUT +
                    L1_OUT * L2_OUT + L2_OUT +
                    L2_OUT * EMBED_DIM + EMBED_DIM)


class TripletEmbedder(nn.Module):
    """100 → 128 → 64 → 32, L2-normalized output."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEAT_DIM, L1_OUT)
        self.fc2 = nn.Linear(L1_OUT, L2_OUT)
        self.fc3 = nn.Linear(L2_OUT, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return F.normalize(h, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(data_dir: Path):
    """Load features.bin + labels.jsonl, group by character.

    Returns:
        char_data: dict[str, list[(font_key, np.ndarray)]]
            Maps character string → list of (font_key, 100-dim feature vector)
    """
    manifest_path = data_dir / "manifest.json"
    features_path = data_dir / "features.bin"
    labels_path = data_dir / "labels.jsonl"

    for p in [manifest_path, features_path, labels_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run gen_training_data first.", file=sys.stderr)
            sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_samples = manifest["n_samples"]
    feat_dim = manifest["feat_dim"]
    assert feat_dim == FEAT_DIM, f"Expected feat_dim={FEAT_DIM}, got {feat_dim}"

    print(f"Loading {n_samples} samples from {data_dir}...")

    # Read features
    features = np.fromfile(str(features_path), dtype=np.float32)
    features = features.reshape(n_samples, feat_dim)

    # Read labels
    labels = []
    with open(labels_path) as f:
        for line in f:
            labels.append(json.loads(line))

    assert len(labels) == n_samples

    # Group by character
    char_data: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for i, label in enumerate(labels):
        char_str = label["char_str"]
        font_key = label["font_key"]
        char_data[char_str].append((font_key, features[i]))

    print(f"  {len(char_data)} unique characters")
    for ch, samples in sorted(char_data.items(), key=lambda x: -len(x[1]))[:5]:
        n_fonts = len(set(fk for fk, _ in samples))
        print(f"    '{ch}': {len(samples)} samples, {n_fonts} fonts")

    return char_data


# ---------------------------------------------------------------------------
# Triplet mining
# ---------------------------------------------------------------------------

def make_triplets(embeddings: torch.Tensor, font_ids: torch.Tensor,
                  margin: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semi-hard negative mining within a batch.

    For each anchor, find the hardest positive (same font, farthest away)
    and semi-hard negative (different font, closer than positive + margin
    but farther than anchor).
    """
    n = embeddings.size(0)
    # Pairwise squared distances
    dists = torch.cdist(embeddings, embeddings, p=2).pow(2)

    anchors, positives, negatives = [], [], []

    for i in range(n):
        font_i = font_ids[i]
        pos_mask = (font_ids == font_i)
        neg_mask = ~pos_mask
        pos_mask[i] = False  # exclude self

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Hardest positive: same font, max distance
        pos_dists = dists[i].clone()
        pos_dists[~pos_mask] = -1
        hardest_pos_idx = pos_dists.argmax()
        d_ap = dists[i, hardest_pos_idx]

        # Semi-hard negatives: different font, d_an > d_ap but d_an < d_ap + margin
        neg_dists = dists[i].clone()
        neg_dists[~neg_mask] = float('inf')

        semi_hard = neg_mask & (neg_dists > d_ap) & (neg_dists < d_ap + margin)
        if semi_hard.any():
            # Pick the hardest semi-hard (closest to anchor among semi-hards)
            sh_dists = neg_dists.clone()
            sh_dists[~semi_hard] = float('inf')
            neg_idx = sh_dists.argmin()
        else:
            # Fall back to hardest negative (closest different-font sample)
            neg_idx = neg_dists.argmin()

        anchors.append(i)
        positives.append(hardest_pos_idx.item())
        negatives.append(neg_idx.item())

    if not anchors:
        return None, None, None

    return (torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives))


# ---------------------------------------------------------------------------
# Training loop for one character
# ---------------------------------------------------------------------------

def train_one_char(char_str: str, samples: list[tuple[str, np.ndarray]],
                   epochs: int, lr: float, margin: float, batch_size: int,
                   device: torch.device, verbose: bool = False) -> TripletEmbedder:
    """Train a triplet embedding network for one character."""

    # Build font_key → integer id mapping
    font_keys = sorted(set(fk for fk, _ in samples))
    fk_to_id = {fk: i for i, fk in enumerate(font_keys)}
    n_fonts = len(font_keys)

    # Tensors
    features = torch.tensor(np.array([feat for _, feat in samples]), dtype=torch.float32)
    font_ids = torch.tensor([fk_to_id[fk] for fk, _ in samples], dtype=torch.long)

    n_samples = features.size(0)

    if n_fonts < 2:
        if verbose:
            print(f"  '{char_str}': only {n_fonts} font, skipping training (identity model)")
        model = TripletEmbedder()
        return model

    model = TripletEmbedder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    features = features.to(device)
    font_ids = font_ids.to(device)

    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        features_shuffled = features[perm]
        font_ids_shuffled = font_ids[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_feat = features_shuffled[start:end]
            batch_fonts = font_ids_shuffled[start:end]

            embeddings = model(batch_feat)

            a_idx, p_idx, n_idx = make_triplets(embeddings, batch_fonts, margin)
            if a_idx is None:
                continue

            loss = triplet_loss(embeddings[a_idx], embeddings[p_idx], embeddings[n_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
        else:
            avg_loss = 0.0

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch+1}")
                break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_weights(models: dict[str, TripletEmbedder], output_path: Path):
    """Export per-glyph models to binary format.

    Format:
        magic: b"TRIP" (4 bytes)
        version: u32 LE (1)
        n_chars: u32 LE
        Per character:
            char_code: u32 LE
            W1: 100×128 f32 LE (row-major)
            b1: 128 f32 LE
            W2: 128×64 f32 LE
            b2: 64 f32 LE
            W3: 64×32 f32 LE
            b3: 32 f32 LE
    """
    with open(output_path, 'wb') as f:
        # Header
        f.write(b"TRIP")
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(models)))

        for char_str, model in sorted(models.items()):
            # Unicode codepoint
            codepoint = ord(char_str)
            f.write(struct.pack('<I', codepoint))

            # Extract weights in the order the Rust side expects
            sd = model.state_dict()
            # fc1.weight is [128, 100] in PyTorch (out×in) — Rust expects row-major [100, 128] (in×out)
            w1 = sd['fc1.weight'].T.contiguous().cpu().numpy()  # [100, 128]
            b1 = sd['fc1.bias'].cpu().numpy()                   # [128]
            w2 = sd['fc2.weight'].T.contiguous().cpu().numpy()  # [128, 64]
            b2 = sd['fc2.bias'].cpu().numpy()                   # [64]
            w3 = sd['fc3.weight'].T.contiguous().cpu().numpy()  # [64, 32]
            b3 = sd['fc3.bias'].cpu().numpy()                   # [32]

            for arr in [w1, b1, w2, b2, w3, b3]:
                f.write(arr.astype(np.float32).tobytes())

    total_bytes = output_path.stat().st_size
    print(f"Exported {len(models)} per-glyph models to {output_path} ({total_bytes:,} bytes)")
    expected = 12 + len(models) * (4 + PARAMS_PER_MODEL * 4)  # header + per-char
    assert total_bytes == expected, f"Size mismatch: {total_bytes} vs expected {expected}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: TripletEmbedder, features: torch.Tensor,
                   font_ids: torch.Tensor, device: torch.device) -> float:
    """Compute top-1 nearest-neighbor accuracy for a glyph model.

    For each sample, find the nearest neighbor (excluding self) and check
    if it's the same font.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(features.to(device))
        dists = torch.cdist(embeddings, embeddings, p=2)
        # Exclude self
        dists.fill_diagonal_(float('inf'))
        nearest = dists.argmin(dim=1)
        correct = (font_ids[nearest.cpu()] == font_ids).float().mean().item()
    return correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train per-glyph triplet embedding networks")
    parser.add_argument("data_dir", type=Path, help="Path to gen_training_data output directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("triplet-weights.bin"),
                        help="Output binary weights file (default: triplet-weights.bin)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per glyph (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet margin (default: 0.3)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch loss")
    parser.add_argument("--chars", type=str, default=None,
                        help="Comma-separated chars to train (default: all)")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")

    char_data = load_training_data(args.data_dir)

    # Filter characters if requested
    if args.chars:
        selected = set(args.chars.split(","))
        char_data = {k: v for k, v in char_data.items() if k in selected}
        print(f"Filtered to {len(char_data)} characters: {sorted(char_data.keys())}")

    models: dict[str, TripletEmbedder] = {}
    accuracies: dict[str, float] = {}

    for i, (char_str, samples) in enumerate(sorted(char_data.items())):
        font_keys = sorted(set(fk for fk, _ in samples))
        n_fonts = len(font_keys)
        print(f"[{i+1}/{len(char_data)}] '{char_str}' (U+{ord(char_str):04X}): "
              f"{len(samples)} samples, {n_fonts} fonts", end="", flush=True)

        model = train_one_char(
            char_str, samples,
            epochs=args.epochs, lr=args.lr, margin=args.margin,
            batch_size=args.batch_size, device=device, verbose=args.verbose,
        )
        models[char_str] = model

        # Evaluate
        fk_to_id = {fk: i for i, fk in enumerate(font_keys)}
        features = torch.tensor(np.array([feat for _, feat in samples]), dtype=torch.float32)
        font_ids = torch.tensor([fk_to_id[fk] for fk, _ in samples], dtype=torch.long)

        if n_fonts >= 2:
            acc = evaluate_model(model, features, font_ids, device)
            accuracies[char_str] = acc
            print(f" → {acc:.1%} nn-accuracy")
        else:
            print(f" → single font, skipped eval")

    # Summary
    if accuracies:
        mean_acc = np.mean(list(accuracies.values()))
        worst_chars = sorted(accuracies.items(), key=lambda x: x[1])[:10]
        print(f"\nMean nn-accuracy: {mean_acc:.1%}")
        print(f"Worst 10 characters:")
        for ch, acc in worst_chars:
            print(f"  '{ch}' (U+{ord(ch):04X}): {acc:.1%}")

    export_weights(models, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
