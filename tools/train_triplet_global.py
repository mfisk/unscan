#!/usr/bin/env python3
"""
Global triplet network trainer for unscan font classification.

Trains a SINGLE embedding network on ALL characters together, unlike the
per-glyph trainer (train_triplet.py) which trains one network per character.

The identity for triplet mining is `font_key` — samples from the same font
(regardless of character) are pulled together.  The resulting embedding
space captures both font and character identity, so nearest-neighbor search
recovers both which character and which font match best.

Architecture: 100 → 128 (ReLU) → 64 (ReLU) → 32 → L2-normalize
Loss: triplet margin loss with semi-hard negative mining

Usage:
    python3 tools/train_triplet_global.py training-data/ -o global-triplet-weights.bin
    python3 tools/train_triplet_global.py training-data/ -o global-triplet-weights.bin --epochs 80

Binary output format (magic b"TRPG"):
    magic:   b"TRPG" (4 bytes)
    version: u32 LE (1)
    W1: 100×128 f32 LE (row-major, in×out)
    b1: 128 f32 LE
    W2: 128×64 f32 LE
    b2: 64 f32 LE
    W3: 64×32 f32 LE
    b3: 32 f32 LE
    Total: 8 + 23,264 × 4 = 93,064 bytes
"""

import argparse
import json
import struct
import sys
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
    """Load features.bin + labels.jsonl as a flat list.

    Returns:
        samples: list[(font_key, np.ndarray)]
        n_fonts: int
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

    # Read labels — we only need font_key
    font_keys_list = []
    with open(labels_path) as f:
        for line in f:
            label = json.loads(line)
            font_keys_list.append(label["font_key"])

    assert len(font_keys_list) == n_samples

    unique_fonts = sorted(set(font_keys_list))
    n_fonts = len(unique_fonts)

    samples = [(fk, features[i]) for i, fk in enumerate(font_keys_list)]

    print(f"  {n_samples} samples, {n_fonts} unique fonts, {manifest['n_chars']} characters")

    return samples, n_fonts


# ---------------------------------------------------------------------------
# Triplet mining
# ---------------------------------------------------------------------------

def make_triplets(embeddings: torch.Tensor, font_ids: torch.Tensor,
                  margin: float = 0.2):
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
            sh_dists = neg_dists.clone()
            sh_dists[~semi_hard] = float('inf')
            neg_idx = sh_dists.argmin()
        else:
            # Fall back to hardest negative
            neg_idx = neg_dists.argmin()

        anchors.append(i)
        positives.append(hardest_pos_idx.item())
        negatives.append(neg_idx.item())

    if not anchors:
        return None, None, None

    return (torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_global(samples, epochs: int, lr: float, margin: float,
                 batch_size: int, device: torch.device,
                 verbose: bool = False) -> TripletEmbedder:
    """Train a single global triplet embedding network on all characters."""

    # Build font_key → integer id mapping
    font_keys = sorted(set(fk for fk, _ in samples))
    fk_to_id = {fk: i for i, fk in enumerate(font_keys)}
    n_fonts = len(font_keys)

    print(f"Training global model: {len(samples)} samples, {n_fonts} fonts")

    if n_fonts < 2:
        print("WARNING: fewer than 2 fonts, returning untrained model")
        return TripletEmbedder()

    # Tensors
    features = torch.tensor(np.array([feat for _, feat in samples]), dtype=torch.float32)
    font_ids = torch.tensor([fk_to_id[fk] for fk, _ in samples], dtype=torch.long)
    n_samples = features.size(0)

    model = TripletEmbedder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    features = features.to(device)
    font_ids = font_ids.to(device)

    best_loss = float('inf')
    patience_counter = 0
    patience = 15

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

            # Need enough diversity in the batch for triplet mining
            unique_in_batch = batch_fonts.unique().numel()
            if unique_in_batch < 2:
                continue

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

        if (epoch + 1) % 5 == 0 or verbose:
            print(f"  epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({n_batches} batches)")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_weights(model: TripletEmbedder, output_path: Path):
    """Export single global model to TRPG binary format.

    Format:
        magic:   b"TRPG" (4 bytes)
        version: u32 LE (1)
        W1: 100×128 f32 LE (row-major, in×out)
        b1: 128 f32 LE
        W2: 128×64 f32 LE
        b2: 64 f32 LE
        W3: 64×32 f32 LE
        b3: 32 f32 LE
    """
    with open(output_path, 'wb') as f:
        # Header
        f.write(b"TRPG")
        f.write(struct.pack('<I', 1))  # version

        # Extract weights — transpose from PyTorch [out, in] to Rust [in, out]
        sd = model.state_dict()
        w1 = sd['fc1.weight'].T.contiguous().cpu().numpy()  # [100, 128]
        b1 = sd['fc1.bias'].cpu().numpy()                   # [128]
        w2 = sd['fc2.weight'].T.contiguous().cpu().numpy()  # [128, 64]
        b2 = sd['fc2.bias'].cpu().numpy()                   # [64]
        w3 = sd['fc3.weight'].T.contiguous().cpu().numpy()  # [64, 32]
        b3 = sd['fc3.bias'].cpu().numpy()                   # [32]

        for arr in [w1, b1, w2, b2, w3, b3]:
            f.write(arr.astype(np.float32).tobytes())

    total_bytes = output_path.stat().st_size
    expected = 8 + PARAMS_PER_MODEL * 4  # header + weights
    print(f"Exported global model to {output_path} ({total_bytes:,} bytes)")
    assert total_bytes == expected, f"Size mismatch: {total_bytes} vs expected {expected}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: TripletEmbedder, features: torch.Tensor,
                   font_ids: torch.Tensor, device: torch.device) -> float:
    """Compute top-1 nearest-neighbor accuracy (same-font retrieval).

    For each sample, find the nearest neighbor (excluding self) and check
    if it's the same font.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(features.to(device))
        dists = torch.cdist(embeddings, embeddings, p=2)
        dists.fill_diagonal_(float('inf'))
        nearest = dists.argmin(dim=1)
        correct = (font_ids[nearest.cpu()] == font_ids).float().mean().item()
    return correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a single global triplet embedding network (all characters)")
    parser.add_argument("data_dir", type=Path,
                        help="Path to gen_training_data output directory")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("global-triplet-weights.bin"),
                        help="Output binary weights file (default: global-triplet-weights.bin)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Max training epochs (default: 80)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet margin (default: 0.3)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-epoch loss")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total samples (for quick testing)")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")

    samples, n_fonts = load_training_data(args.data_dir)

    if args.max_samples and len(samples) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(samples)
        samples = samples[:args.max_samples]
        print(f"Subsampled to {len(samples)} samples")

    model = train_global(
        samples,
        epochs=args.epochs, lr=args.lr, margin=args.margin,
        batch_size=args.batch_size, device=device, verbose=args.verbose,
    )

    # Final evaluation
    font_keys = sorted(set(fk for fk, _ in samples))
    fk_to_id = {fk: i for i, fk in enumerate(font_keys)}
    features = torch.tensor(np.array([feat for _, feat in samples]), dtype=torch.float32)
    font_ids = torch.tensor([fk_to_id[fk] for fk, _ in samples], dtype=torch.long)

    acc = evaluate_model(model, features, font_ids, device)
    print(f"\nFinal nn-accuracy (same-font retrieval): {acc:.1%}")

    export_weights(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
