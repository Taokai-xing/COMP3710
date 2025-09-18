import os, math, argparse, random
from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1) Dataset for grayscale PNG slices (values normalized to [0,1])
# ------------------------------------------------------------
class PNGSlices(Dataset):
    """
    Reads grayscale slices from a folder.
    Supports: *.png, *.nii.png, *.jpg, *.jpeg (case-insensitive)
    Recursively scans subfolders.
    """
    def __init__(self, root_dir: str, img_size: int = 128):
        self.root = root_dir
        self.img_size = img_size

        # Collect files recursively with multiple patterns
        patterns = ["*.png", "*.nii.png", "*.jpg", "*.jpeg", "*.PNG", "*.NII.PNG", "*.JPG", "*.JPEG"]
        paths = []
        for pat in patterns:
            paths += glob(os.path.join(root_dir, "**", pat), recursive=True)

        # De-duplicate and sort
        self.paths = sorted({os.path.normpath(p) for p in paths})

        if not self.paths:
            raise FileNotFoundError(
                f"No image files found in: {root_dir}\n"
                f"Tried patterns: {patterns}\n"
                f"Tip: check whether your files are '.nii.png' or in subfolders."
            )

        # Optional: show a small sample for sanity check
        print(f"[PNGSlices] Found {len(self.paths)} files under: {root_dir}")
        for p in self.paths[:3]:
            print("  e.g.", p)

    def __len__(self):
        return len(self.paths)

    def _load_img_gray01(self, path: str) -> np.ndarray:
        # Treat everything as a regular PNG/JPG (even if name contains .nii.png)
        img = Image.open(path).convert("L")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return (np.array(img, dtype=np.float32) / 255.0)

    def __getitem__(self, idx: int):
        x = self._load_img_gray01(self.paths[idx])  # [H,W] in [0,1]
        x = torch.from_numpy(x)[None, ...]          # [1,H,W]
        return x

# ------------------------------------------------------------
# 2) A small VAE for 1xHxW images
# ------------------------------------------------------------
class Encoder(nn.Module):
    """
    Convolutional encoder that maps 1xHxW -> (mu, logvar) with latent_dim.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # Downsampling convs
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),# H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),# H/16
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_mu    = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar= nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = self.flatten(h)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Deconvolutional decoder that maps z -> 1xHxW.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 64->128
            nn.Sigmoid(),                          # output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)                 # [N,256*8*8]
        h = h.view(-1, 256, 8, 8)      # [N,256,8,8]
        x = self.deconv(h)             # [N,1,128,128]
        return x


class VAE(nn.Module):
    """
    Standard VAE: x -> Encoder -> (mu, logvar) -> sample z -> Decoder -> x_hat
    Loss = recon_loss (BCE) + KL divergence
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


# ------------------------------------------------------------
# 3) Utils for training, saving, and visualization
# ------------------------------------------------------------
def vae_loss(x, x_hat, mu, logvar) -> torch.Tensor:
    """
    Binary cross-entropy reconstruction + KL divergence to N(0,I).
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / N
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon.detach(), kl.detach()

@torch.no_grad()
def save_recon_grid(x: torch.Tensor, x_hat: torch.Tensor, out_path: str, max_n: int = 8):
    """
    Save a side-by-side grid of original vs reconstruction for the first max_n samples.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = min(max_n, x.size(0))
    x  = x[:n].cpu().numpy()      # [n,1,H,W]
    xr = x_hat[:n].cpu().numpy()  # [n,1,H,W]
    # Construct a 2-row grid: row0 originals, row1 reconstructions
    row = 2
    col = n
    H, W = x.shape[2], x.shape[3]
    canvas = np.zeros((row*H, col*W), dtype=np.float32)
    for i in range(n):
        canvas[0:H, i*W:(i+1)*W] = x[i,0]
        canvas[H:2*H, i*W:(i+1)*W] = xr[i,0]
    img = Image.fromarray((canvas*255).astype(np.uint8))
    img.save(out_path)
    print(f"[Saved] Reconstructions -> {out_path}")

@torch.no_grad()
def save_manifold_2d(decoder: Decoder, out_path: str, grid: int = 20, span: float = 2.5, img_size: int = 128):
    """
    If latent_dim=2, sample a grid over z1/z2 in [-span, span] and decode to an image manifold.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lin = np.linspace(-span, span, grid)
    canvas = np.zeros((grid*img_size, grid*img_size), dtype=np.float32)
    device = next(decoder.parameters()).device
    for i, z1 in enumerate(lin):
        for j, z2 in enumerate(lin):
            z = torch.tensor([[z1, z2]], dtype=torch.float32, device=device) # [1,2]
            x = decoder(z)   # [1,1,H,W]
            tile = x[0,0].detach().cpu().numpy()
            canvas[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size] = tile
    img = Image.fromarray((canvas*255).astype(np.uint8))
    img.save(out_path)
    print(f"[Saved] 2D manifold -> {out_path}")

@torch.no_grad()
def save_umap_scatter(mu_all: np.ndarray, out_path: str):
    """
    If latent_dim>2, reduce mu to 2D with UMAP and save a scatter plot.
    """
    try:
        import umap
    except Exception as e:
        print("[Warn] umap-learn not installed. Install via: pip install umap-learn")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
    emb = reducer.fit_transform(mu_all)  # [N,2]
    # Simple scatter via PIL (no matplotlib dependency)
    # Normalize to [0,1] and draw points on a white canvas.
    xy = (emb - emb.min(0)) / (emb.ptp(0) + 1e-8)
    H = W = 800
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    for x, y in xy:
        r, c = int((1-y)*H*0.95+H*0.025), int(x*W*0.95+W*0.025)
        canvas[r-1:r+2, c-1:c+2] = [0, 114, 189]  # small dot
    Image.fromarray(canvas).save(out_path)
    print(f"[Saved] UMAP scatter -> {out_path}")


# ------------------------------------------------------------
# 4) Training loop
# ------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # Datasets
    train_ds = PNGSlices(args.train_dir, img_size=args.img_size)
    val_ds   = PNGSlices(args.val_dir,   img_size=args.img_size)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    vae = VAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        vae.train()
        tr_loss = tr_recon = tr_kl = 0.0
        for x in train_ld:
            x = x.to(device)
            x_hat, mu, logvar, _ = vae(x)
            loss, rec, kl = vae_loss(x, x_hat, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += float(loss) * x.size(0)
            tr_recon += float(rec) * x.size(0)
            tr_kl += float(kl) * x.size(0)

        ntr = len(train_ds)
        tr_loss /= ntr; tr_recon /= ntr; tr_kl /= ntr

        # Validation
        vae.eval()
        val_loss = val_recon = val_kl = 0.0
        mus = []
        with torch.no_grad():
            for x in val_ld:
                x = x.to(device)
                x_hat, mu, logvar, _ = vae(x)
                loss, rec, kl = vae_loss(x, x_hat, mu, logvar)
                val_loss += float(loss) * x.size(0)
                val_recon += float(rec) * x.size(0)
                val_kl += float(kl) * x.size(0)
                mus.append(mu.detach().cpu().numpy())
        nval = len(val_ds)
        val_loss /= nval; val_recon /= nval; val_kl /= nval

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} (rec {tr_recon:.4f}, kl {tr_kl:.4f}) "
              f"| val {val_loss:.4f} (rec {val_recon:.4f}, kl {val_kl:.4f})")

        # Save sample reconstructions from the first val batch
        # (take one batch again to save visuals)
        x = next(iter(val_ld)).to(device)
        x_hat, mu, logvar, z = vae(x)
        save_recon_grid(x, x_hat, out_path=os.path.join(args.out_dir, f"recon_ep{epoch:03d}.png"))

        # Save 2D manifold (if latent_dim=2) or UMAP scatter (if >2)
        if args.latent_dim == 2:
            save_manifold_2d(vae.decoder, out_path=os.path.join(args.out_dir, f"manifold_ep{epoch:03d}.png"),
                             grid=20, span=2.5, img_size=args.img_size)
        else:
            mu_all = np.concatenate(mus, axis=0)  # [N, latent_dim]
            save_umap_scatter(mu_all, out_path=os.path.join(args.out_dir, f"umap_ep{epoch:03d}.png"))

        # Save best checkpoint by val_loss
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.ckpt_dir, "vae_oasis_png.pth")
            torch.save({"model": vae.state_dict(),
                        "latent_dim": args.latent_dim,
                        "img_size": args.img_size}, ckpt_path)
            print(f"[Saved] best ckpt -> {ckpt_path} (val {best_val:.4f})")


# ------------------------------------------------------------
# 5) CLI
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train a VAE on OASIS PNG slices and visualize manifold/UMAP.")
    ap.add_argument("--train_dir", type=str, required=True, help="Folder of training PNGs")
    ap.add_argument("--val_dir",   type=str, required=True, help="Folder of validation PNGs")
    ap.add_argument("--img_size",  type=int, default=128, help="Resize to square HxW")
    ap.add_argument("--latent_dim",type=int, default=2, help="2 for manifold; >2 will use UMAP")
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--ckpt_dir",  type=str, default="checkpoints")
    ap.add_argument("--out_dir",   type=str, default="results/vae")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
