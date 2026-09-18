"""
Training script for TrajectoryTransformer.

Metrics (standard in the trajectory-prediction literature):
  ADE (Average Displacement Error) - mean L2 error over all predicted steps
  FDE (Final Displacement Error)   - L2 error at the last predicted step only

Run:
    python train.py --epochs 20 --data synthetic
    python train.py --epochs 20 --data ethucy --data_dir data/eth/train
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import TrajectoryTransformer
from dataset import SyntheticCrowdDataset, ETHUCYDataset, make_relative


def masked_mse_loss(pred, target, mask):
    # pred, target: (B, N, T, 2); mask: (B, N)
    diff = (pred - target) ** 2
    diff = diff.sum(-1)  # (B, N, T)
    mask_exp = mask.unsqueeze(-1).float()
    diff = diff * mask_exp
    return diff.sum() / (mask_exp.sum() * diff.shape[-1] + 1e-8)


@torch.no_grad()
def compute_ade_fde(pred, target, mask):
    # pred, target: (B, N, T, 2) absolute coords; mask: (B, N)
    err = torch.norm(pred - target, dim=-1)  # (B, N, T)
    mask_exp = mask.unsqueeze(-1).float()

    ade = (err * mask_exp).sum() / (mask_exp.sum() * err.shape[-1] + 1e-8)
    fde_err = err[:, :, -1]  # (B, N)
    fde = (fde_err * mask).sum() / (mask.sum() + 1e-8)
    return ade.item(), fde.item()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_ade, total_fde, n_batches = 0.0, 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs, fut, mask in loader:
            obs, fut, mask = obs.to(device), fut.to(device), mask.to(device)

            # normalize: relative to last observed position
            obs_rel, origin = make_relative(obs)
            fut_rel = fut - origin  # keep future in same relative frame

            pred_rel = model(obs_rel, mask)
            loss = masked_mse_loss(pred_rel, fut_rel, mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pred_abs = pred_rel + origin
            ade, fde = compute_ade_fde(pred_abs, fut, mask)

            total_loss += loss.item()
            total_ade += ade
            total_fde += fde
            n_batches += 1

    return total_loss / n_batches, total_ade / n_batches, total_fde / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "ethucy"], default="synthetic")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--max_agents", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.data == "synthetic":
        full_ds = SyntheticCrowdDataset(
            n_scenes=2000,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            max_agents=args.max_agents,
        )
    else:
        if args.data_dir is None:
            raise ValueError("--data_dir is required for --data ethucy")
        full_ds = ETHUCYDataset(
            args.data_dir, obs_len=args.obs_len, pred_len=args.pred_len,
            max_agents=args.max_agents,
        )

    n_val = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TrajectoryTransformer(
        d_model=args.d_model, obs_len=args.obs_len, pred_len=args.pred_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ade = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ade, train_fde = run_epoch(
            model, train_loader, optimizer, device, train=True
        )
        val_loss, val_ade, val_fde = run_epoch(
            model, val_loader, optimizer, device, train=False
        )

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_loss:.4f} ADE {train_ade:.3f} FDE {train_fde:.3f} | "
            f"val_loss {val_loss:.4f} ADE {val_ade:.3f} FDE {val_fde:.3f}"
        )

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), args.save_path)

    print(f"Best val ADE: {best_val_ade:.3f}. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
