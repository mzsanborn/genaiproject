"""
Diffusion (noise-prediction) loss of a trained checkpoint on the train / val / test splits.

    python eval_loss.py --run <exp>/logs/tcga [--device cpu] [--draws 50]

Uses the EMA weights (what sampling uses), eval mode (no dropout), the same
data scaling as training (scaler fit on train), and the training objective
||eps - eps_theta(x_t, t, y)||^2. Each sample is scored at `draws` random
timesteps with fixed seeds, so splits are compared on identical noise draws.
Training never computes a val/test loss; this is how to get one.
"""
import argparse, os, sys, yaml
import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from src.generation.ddim.runners.diffusion import get_beta_schedule, build_model
from src.generation.ddim.models.ema import EMAHelper
from src.generation.ddim.datasets.utils import get_tcga_datasets


def load_config(run):
    cfg = yaml.unsafe_load(open(os.path.join(run, "config.yml")))
    return cfg


@torch.no_grad()
def split_loss(model, X, y, betas, draws, device, seed=0, bs=512):
    a_bar = (1 - betas).cumprod(0)
    T = len(betas)
    g = torch.Generator().manual_seed(seed)
    per_sample = torch.zeros(len(X))
    for _ in range(draws):
        t = torch.randint(0, T, (len(X),), generator=g)
        e = torch.randn(X.shape, generator=g)
        for i in range(0, len(X), bs):
            xb, yb, tb, eb = X[i:i + bs], y[i:i + bs], t[i:i + bs], e[i:i + bs]
            a = a_bar[tb].view(-1, 1)
            xt = xb * a.sqrt() + eb * (1 - a).sqrt()
            out = model(xt.to(device), tb.float().to(device), yb.to(device)).float().cpu()
            per_sample[i:i + bs] += (eb - out).square().sum(1)
    per_sample /= draws
    return per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with ckpt.pth and config.yml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--draws", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.run)
    torch.manual_seed(0)
    model = build_model(cfg).to(args.device)
    states = torch.load(os.path.join(args.run, "ckpt.pth"), map_location=args.device, weights_only=False)
    model.load_state_dict(states[0])
    if cfg.model.ema:
        ema = EMAHelper(cfg.model.ema_rate); ema.register(model); ema.load_state_dict(states[-1]); ema.ema(model)
    model.eval()
    betas = torch.from_numpy(get_beta_schedule(cfg.diffusion.beta_schedule, beta_start=cfg.diffusion.beta_start,
                                               beta_end=cfg.diffusion.beta_end,
                                               num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps)).float()
    cond = getattr(cfg.data, "conditioning", "tissue")
    print(f"checkpoint step {states[3]}, model {cfg.model.model}, conditioning {cond}, "
          f"{args.draws} timestep draws per sample")

    rows = []
    for split in ["val", "test"]:
        train, held = get_tcga_datasets(cfg.data.scaler_type, conditioning=cond, eval_split=split)
        if split == "val":
            Xtr, ytr = train.tensors
            # same size as the held-out splits, fixed subset
            idx = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))[:len(held)]
            rows.append(("train (subset)", split_loss(model, Xtr[idx], ytr[idx], betas, args.draws, args.device)))
        Xh, yh = held.tensors
        rows.append((split, split_loss(model, Xh, yh, betas, args.draws, args.device)))

    G = cfg.data.image_size
    print(f"\n{'split':<16}{'n':>6}{'loss (sum over genes)':>24}{'per-gene MSE':>15}")
    for name, ps in rows:
        m, se = ps.mean().item(), ps.std().item() / np.sqrt(len(ps))
        print(f"{name:<16}{len(ps):>6}{m:>17.2f} ± {se:<5.2f}{m / G:>14.4f}")


if __name__ == "__main__":
    main()
