"""
train_predictor_matched.py

Retrains the trajectory predictor on data generated from crowd_env's
own social-force pedestrian dynamics (see generate_matched_dataset.py),
instead of the original separate synthetic dataset. This fixes the
distribution mismatch that likely caused the "with prediction" RL
policy to underperform the no-prediction baseline.

Run:
    python train_predictor_matched.py --epochs 20

Saves to checkpoint_matched.pt by default.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import TrajectoryTransformer
from generate_matched_dataset import CrowdEnvTrajectoryDataset


def make_relative(traj):
    origin = traj[..., 0:1, :]
    return traj - origin, origin


def masked_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.sum(-1)
    mask_exp = mask.unsqueeze(-1).float()
    diff = diff * mask_exp
    return diff.sum() / (mask_exp.sum() * diff.shape[-1] + 1e-8)


@torch.no_grad()
def compute_ade_fde(pred, target, mask):
    err = torch.norm(pred - target, dim=-1)
    mask_exp = mask.unsqueeze(-1).float()
    ade = (err * mask_exp).sum() / (mask_exp.sum() * err.shape[-1] + 1e-8)
    fde_err = err[:, :, -1]
    fde = (fde_err * mask).sum() / (mask.sum() + 1e-8)
    return ade.item(), fde.item()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_ade, total_fde, n_batches = 0.0, 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs, fut, mask in loader:
            obs, fut, mask = obs.to(device), fut.to(device), mask.to(device)
            obs_rel, origin = make_relative(obs)
            fut_rel = fut - origin

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
    parser.add_argument("--n_scenes", type=int, default=3000)
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--max_agents", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="checkpoint_matched.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Generating {args.n_scenes} scenes from crowd_env's social-force pedestrians...")
    full_ds = CrowdEnvTrajectoryDataset(
        n_scenes=args.n_scenes,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_agents=args.max_agents,
    )
    print("Data generation complete.")

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
        train_loss, train_ade, train_fde = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_ade, val_fde = run_epoch(model, val_loader, optimizer, device, train=False)

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
