"""
train2.py
---------

Fast, resource-aware training and screening entrypoint for the biosafety VAE.

This script is tuned for laptops (e.g. MacBook Air M3) by default: it trains
on a small subsample of the data and uses moderate batch sizes / epochs so
that runs are quick and your machine does not get excessively hot.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

from src.data_processor import GenomicDataProcessor
from src.vae_model import GenomicVAE


def load_seqs_limited(fasta_path: str, max_per_file: Optional[int] = None, min_len: int = 50):
    """
    Stream sequences from a FASTA file, keeping at most ``max_per_file`` records.

    This avoids loading very large files fully into memory and speeds up training.
    """
    sequences: list[str] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        if len(record.seq) < min_len:
            continue
        sequences.append(str(record.seq).upper())
        if max_per_file is not None and len(sequences) >= max_per_file:
            break
    return sequences


def build_device() -> torch.device:
    """
    Choose the best available compute device.

    Preference order: Apple Metal (MPS) → CUDA GPU → CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fast(
    safe_path: str,
    amr_path: str,
    vf_path: str,
    max_per_file: int = 200,
    max_total: int = 300,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 5e-4,
    beta_kl: float = 4.0,
    model_out: str = "models/vae_best_fast.pth",
) -> None:
    """
    Train a VAE quickly on a subsampled dataset.

    The default settings are intentionally conservative so that training completes
    quickly and keeps CPU/GPU usage modest on a MacBook Air–class machine.
    """
    print("[train2] Loading training data (fast mode)...")
    safe_sequences = load_seqs_limited(safe_path, max_per_file)
    amr_sequences = load_seqs_limited(amr_path, max_per_file)
    vf_sequences = load_seqs_limited(vf_path, max_per_file)
    all_sequences = safe_sequences + amr_sequences + vf_sequences
    print(f"[train2] Loaded {len(all_sequences):,} sequences before subsampling")

    if len(all_sequences) == 0:
        raise RuntimeError("No sequences loaded for training. Check input FASTA paths.")

    # Subsample globally to keep training quick
    if len(all_sequences) > max_total:
        sampled_idx = np.random.choice(len(all_sequences), size=max_total, replace=False)
        all_sequences = [all_sequences[i] for i in sampled_idx]
    print(f"[train2] Using {len(all_sequences):,} sequences for training (cap {max_total})")

    processor = GenomicDataProcessor()
    feature_matrix = processor.extract_features(all_sequences)
    dataset = TensorDataset(torch.tensor(feature_matrix))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = build_device()
    print(f"[train2] Training on device: {device}")
    model = GenomicVAE(feature_matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_features)
            recon_loss = ((recon - batch_features) ** 2).sum()
            kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum()
            loss = recon_loss + beta_kl * kl
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        mean_loss = loss_sum / len(all_sequences)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[train2] Epoch {epoch:3d}/{epochs:3d} | Loss: {mean_loss:.4f}")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"[train2] Model saved -> {model_out}")


def screen_fasta(
    fasta_path: str,
    model_path: str = "models/vae_best_fast.pth",
) -> None:
    """
    Screen an input FASTA file using a trained fast VAE model.

    Prints a simple novelty-detection report, using mean+3*std of reconstruction
    errors as the anomaly threshold.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"No trained model found at '{model_file}'. "
            "Run 'python train2.py train-fast' first."
        )

    device = build_device()
    print(f"[train2] Screening {fasta_path} on device {device}...")

    processor = GenomicDataProcessor()
    sequences = load_seqs_limited(fasta_path, max_per_file=None)
    if len(sequences) == 0:
        raise RuntimeError("No sequences loaded for screening. Check input FASTA file.")

    X_test = processor.extract_features(sequences)
    X_test = torch.tensor(X_test).to(device)

    model = GenomicVAE(X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        recon, _, _ = model(X_test)
        errors = ((recon - X_test) ** 2).mean(dim=1).cpu().numpy()

    threshold = float(np.mean(errors) + 3.0 * np.std(errors))
    print("\n[train2] NOVELTY DETECTION REPORT")
    print("-" * 60)
    for idx, err in enumerate(errors):
        status = "THREAT" if err > threshold else "safe"
        print(f"{idx:4d} | Error {err:.6f} -> {status}")
    print(f"\n[train2] {int((errors > threshold).sum())} threats detected above threshold {threshold:.6f}")


def main() -> None:
    """CLI wrapper for fast training and screening."""
    parser = argparse.ArgumentParser(
        description="Fast VAE training and screening for biosafety project (laptop-friendly defaults)."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Fast training mode
    train_parser = subparsers.add_parser("train-fast", help="Train VAE with limited data for speed")
    train_parser.add_argument("--safe", default="data/safe_baseline.fasta", help="Path to safe baseline FASTA")
    train_parser.add_argument("--amr", default="data/known_amr.fasta", help="Path to known AMR FASTA")
    train_parser.add_argument("--vf", default="data/known_vf.fasta", help="Path to known VF FASTA")
    train_parser.add_argument("--max-per-file", type=int, default=200, help="Max sequences per FASTA file")
    train_parser.add_argument("--max-total", type=int, default=300, help="Global cap on training sequences")
    train_parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--beta-kl", type=float, default=4.0, help="KL weight in VAE loss")
    train_parser.add_argument("--model-out", default="models/vae_best_fast.pth", help="Path to save trained model")

    # Screening mode
    screen_parser = subparsers.add_parser("screen", help="Screen FASTA using a trained fast model")
    screen_parser.add_argument("fasta", help="Input FASTA to screen")
    screen_parser.add_argument("--model", default="models/vae_best_fast.pth", help="Path to trained model")

    args = parser.parse_args()

    if args.mode == "train-fast":
        train_fast(
            safe_path=args.safe,
            amr_path=args.amr,
            vf_path=args.vf,
            max_per_file=args.max_per_file,
            max_total=args.max_total,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            model_out=args.model_out,
        )
    elif args.mode == "screen":
        screen_fasta(
            fasta_path=args.fasta,
            model_path=args.model,
        )


if __name__ == "__main__":
    main()


