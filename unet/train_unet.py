

import os, re, random, argparse, math, json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utilities
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # speed up on GPU

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def imread_gray(path: Path):
    # returns PIL.Image in L mode
    return Image.open(str(path)).convert("L")

def to_tensor01(img_pil: Image.Image):
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

def resize_pair(x_pil: Image.Image, y_pil: Image.Image, size: int):
    # bilinear for image, nearest for mask
    x = x_pil.resize((size, size), Image.BILINEAR)
    y = y_pil.resize((size, size), Image.NEAREST)
    return x, y

def mask_to_index(y_pil: Image.Image, classes: int):
    """
    Robustly map PNG mask (0..255) to integer class ids 0..C-1.
    We bucket grayscale into C bins: round( (val/255)*(C-1) ).
    """
    y = np.array(y_pil, dtype=np.float32)
    if y.ndim == 3:  # if RGB, convert to gray
        y = y.mean(axis=2)
    y = np.round((y / 255.0) * (classes - 1)).astype(np.int64)
    y = np.clip(y, 0, classes - 1)
    return torch.from_numpy(y)  # [H,W], long

# ---------------------------
# Dataset (robust pairing)
# ---------------------------
CASE_RE = re.compile(r"(?:^|_)case_(\d+)_slice_(\d+)", re.IGNORECASE)
SEG_RE  = re.compile(r"(?:^|_)seg_(\d+)_slice_(\d+)",  re.IGNORECASE)

def norm_key_from_name(name: str):
    """
    Normalize filename into a comparable key like '###_slice_##'.
    Works for both 'case_123_slice_4.nii.png' and 'seg_123_slice_4.nii.png'
    """
    base = Path(name).name.lower()
    base = base.replace(".nii", "").replace(".png", "")
    m = CASE_RE.search(base) or SEG_RE.search(base)
    if not m:
        return None
    pid, sl = m.group(1), m.group(2)
    return f"{int(pid):03d}_slice_{int(sl)}"

class OASISPNGPaired(Dataset):
    def __init__(self, img_dir, msk_dir, img_size=256, classes=4, raise_on_missing=True):
        self.img_dir = Path(img_dir); self.msk_dir = Path(msk_dir)
        self.img_size = img_size; self.classes = classes
        assert self.img_dir.exists(), f"Images dir not found: {self.img_dir}"
        assert self.msk_dir.exists(), f"Masks dir not found: {self.msk_dir}"

        # index masks by normalized key
        mask_files = [p for p in self.msk_dir.iterdir() if p.suffix.lower()==".png"]
        key_to_mask = {}
        for p in mask_files:
            k = norm_key_from_name(p.name)
            if k: key_to_mask[k] = p

        pairs = []
        # collect images and find matching masks
        img_files = [p for p in self.img_dir.iterdir() if p.suffix.lower()==".png"]
        for p in img_files:
            k = norm_key_from_name(p.name)
            if not k: continue
            m = key_to_mask.get(k, None)
            if m is not None and m.exists():
                pairs.append((p, m))

        if len(pairs) == 0:
            # show a few examples to help debugging
            examples = [p.name for p in img_files[:10]]
            raise FileNotFoundError(
                f"No (image,mask) pairs found. Check your folders:\n"
                f"  imgs={self.img_dir}\n  msks={self.msk_dir}\n"
                f"Examples of images seen: {examples}"
            )

        # if many images had no masks, optionally warn or raise
        if raise_on_missing:
            missing = len(img_files) - len(pairs)
            if missing > 0:
                print(f"[INFO] {missing} image(s) had no matching mask and were skipped.")

        self.pairs = pairs
        print(f"[OK] Found {len(self.pairs)} (image,mask) pairs in\n  {self.img_dir}\n  {self.msk_dir}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        x_pil = imread_gray(img_path)
        y_pil = Image.open(str(msk_path))  # keep palette & NEAREST later

        x_pil, y_pil = resize_pair(x_pil, y_pil, self.img_size)
        x = to_tensor01(x_pil)                          # [1,H,W], float in [0,1]
        y = mask_to_index(y_pil, self.classes).long()   # [H,W], long in [0..C-1]
        return x, y

# ---------------------------
# UNet (standard, channel-safe)
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    """
    Up block that takes 'in_ch' (from deeper layer), upsamples to in_ch//2,
    concatenates with 'skip' (skip_ch), then DoubleConv to 'out_ch'.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes mismatch (can happen due to odd/even)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, 1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, base=64):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(n_channels, base)        # 1   -> 64
        self.down1 = Down(base,     base*2)              # 64  -> 128
        self.down2 = Down(base*2,   base*4)              # 128 -> 256
        self.down3 = Down(base*4,   base*8)              # 256 -> 512
        self.down4 = Down(base*8,   base*16)             # 512 -> 1024

        # Decoder（注意这里传入的是 (in_ch, skip_ch, out_ch)）
        self.up1 = Up(base*16, base*8,  base*8)          # 1024 -> up 512, cat 512 -> 1024 -> 512
        self.up2 = Up(base*8,  base*4,  base*4)          # 512  -> up 256, cat 256 -> 512  -> 256
        self.up3 = Up(base*4,  base*2,  base*2)          # 256  -> up 128, cat 128 -> 256  -> 128
        self.up4 = Up(base*2,  base,    base)            # 128  -> up 64,  cat 64  -> 128  -> 64
        self.outc = OutConv(base, n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        x  = self.up1(x5, x4)  # 512
        x  = self.up2(x,  x3)  # 256
        x  = self.up3(x,  x2)  # 128
        x  = self.up4(x,  x1)  # 64
        return self.outc(x)    # [N,C,H,W]
# ---------------------------
# Dice (per-class) + CE
# ---------------------------
@torch.no_grad()
def dice_per_class_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6):
    # logits: [N,C,H,W], target: [N,H,W] (0..C-1)
    N, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()  # [N,C,H,W]
    inter = (probs * onehot).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + onehot.sum(dim=(0,2,3))
    dice = (2*inter + eps) / (denom + eps)  # [C]
    return dice

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        d = dice_per_class_from_logits(logits, target, eps=self.eps)
        return 1.0 - d.mean()

# ---------------------------
# Train / Val
# ---------------------------
def train(cfg):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # datasets
    train_ds = OASISPNGPaired(cfg.img_train, cfg.msk_train, img_size=cfg.img_size, classes=cfg.classes, raise_on_missing=True)
    val_ds   = OASISPNGPaired(cfg.img_val,   cfg.msk_val,   img_size=cfg.img_size, classes=cfg.classes, raise_on_missing=False)

    train_ld = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,  num_workers=cfg.workers, pin_memory=(device.type=='cuda'))
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=(device.type=='cuda'))

    # model / opt
    model = UNet(n_channels=1, n_classes=cfg.classes, base=64).to(device)
    ce = nn.CrossEntropyLoss()
    dsc_loss = SoftDiceLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)  # no verbose flag -> avoids warning
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type=='cuda'))

    # ckpt
    ckpt_path = Path(cfg.ckpt)
    ensure_dir(ckpt_path.parent)
    best_mean = -1.0

    for epoch in range(1, cfg.epochs+1):
        model.train()
        tr_loss_accum = 0.0
        n_batches = 0

        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type=='cuda')):
                logits = model(x)
                loss = ce(logits, y) + dsc_loss(logits, y)

            scaler.scale(loss).to(dtype=torch.float32)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss_accum += loss.item()
            n_batches += 1

        sched.step()
        tr_loss = tr_loss_accum / max(1, n_batches)

        # validation
        model.eval()
        ce_accum, dsc_accum = 0.0, torch.zeros(cfg.classes, device=device)
        m = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for x, y in val_ld:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                ce_accum += ce(logits, y).item()
                dsc = dice_per_class_from_logits(logits, y)  # [C]
                dsc_accum += dsc
                m += 1

        val_ce = ce_accum / max(1, m)
        per_class = (dsc_accum / max(1, m)).detach().cpu().numpy()
        mean_all = float(per_class.mean())
        mean_fg  = float(per_class[1:].mean()) if cfg.classes > 1 else mean_all

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | valCE {val_ce:.4f} | "
              f"meanDice(all) {mean_all:.4f} | per-class {np.round(per_class,4)}")

        # save best on foreground mean (stricter), fallback to all
        score = mean_fg if cfg.classes > 1 else mean_all
        if score > best_mean:
            best_mean = score
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "classes": cfg.classes,
                "img_size": cfg.img_size,
                "mean_dice_all": mean_all,
                "mean_dice_fg": mean_fg,
                "config": vars(cfg)
            }, str(ckpt_path))
            print(f"[SAVE] Best mean Dice (fg): {best_mean:.4f} -> {ckpt_path}")

    print(f"Done. Best mean Dice (fg): {best_mean:.4f}")

# ---------------------------
# Defaults & CLI
# ---------------------------
def build_parser_with_defaults():
    parser = argparse.ArgumentParser(description="Train UNet on OASIS PNG slices (categorical segmentation).", add_help=True)
    # DEFAULTS set to your environment so script runs out-of-box
    parser.add_argument("--img-train", default="D:/3710/keras_png_slices_data/keras_png_slices_train", type=str)
    parser.add_argument("--msk-train", default="D:/3710/keras_png_slices_data/keras_png_slices_seg_train", type=str)
    parser.add_argument("--img-val",   default="D:/3710/keras_png_slices_data/keras_png_slices_validate", type=str)
    parser.add_argument("--msk-val",   default="D:/3710/keras_png_slices_data/keras_png_slices_seg_validate", type=str)

    parser.add_argument("--classes",   default=4, type=int)
    parser.add_argument("--img-size",  default=256, type=int)
    parser.add_argument("--epochs",    default=40, type=int)
    parser.add_argument("--batch",     default=8, type=int)
    parser.add_argument("--lr",        default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--workers",   default=0, type=int)  # Windows-safe
    parser.add_argument("--amp",       action="store_true", default=True, help="Enable mixed precision (default: on)")
    parser.add_argument("--ckpt",      default="checkpoints/unet_oasis_png.pth", type=str)
    parser.add_argument("--seed",      default=42, type=int)
    return parser

def main():
    parser = build_parser_with_defaults()
    args = parser.parse_args([])
    # If user actually passed CLI flags, argparse won't see them when parse_args([]).
    # So detect and re-parse real CLI when script is invoked from terminal.
    import sys
    if len(sys.argv) > 1:
        args = parser.parse_args()

    # expand and normalize paths
    args.img_train = str(Path(args.img_train))
    args.msk_train = str(Path(args.msk_train))
    args.img_val   = str(Path(args.img_val))
    args.msk_val   = str(Path(args.msk_val))
    args.ckpt      = str(Path(args.ckpt))

    train(args)

if __name__ == "__main__":
    main()
