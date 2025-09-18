# eval_unet_on_test.py
# Self-contained evaluator: includes a clean UNet so it won't hit `self.double(...)` mistakes.
import os
import argparse
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# A clean UNet implementation
# ----------------------------
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
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

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscale, concatenate skip, then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch = (features_from_encoder + features_from_up)
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad/crop to handle odd sizes
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """UNet with categorical (one-hot) output via logits [N,C,H,W]"""
    def __init__(self, n_classes=4, base=64, in_ch=1):
        super().__init__()
        self.inc  = DoubleConv(in_ch, base)          # 1 -> 64
        self.down1 = Down(base, base*2)              # 64 -> 128
        self.down2 = Down(base*2, base*4)            # 128 -> 256
        self.down3 = Down(base*4, base*8)            # 256 -> 512
        self.down4 = Down(base*8, base*8)            # 512 -> 512

        self.up1 = Up(base*16, base*4)               # (512+512)->256
        self.up2 = Up(base*8,  base*2)               # (256+256)->128
        self.up3 = Up(base*4,  base)                 # (128+128)->64
        self.up4 = Up(base*2,  base)                 # (64+64)->64
        self.outc = OutConv(base, n_classes)         # 64 -> C

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)  # logits


# ----------------------------
# Dataset that pairs images/masks by normalized stem
# ----------------------------
TOKENS_TO_STRIP = ('.nii', '_seg', '_mask', '_segmentation')

def _norm_stem(name: str) -> str:
    stem = os.path.splitext(name)[0]
    for t in TOKENS_TO_STRIP:
        stem = stem.replace(t, '')
    return stem

class OASISPNGDataset(Dataset):
    def __init__(self, img_dir: str, msk_dir: str, img_size: int = 160):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.img_size = img_size

        imgs = sorted(sum([glob.glob(os.path.join(img_dir, pat)) for pat in
                           ['*.png','*.nii.png','*.PNG','*.NII.PNG']], []))
        msks = sorted(sum([glob.glob(os.path.join(msk_dir, pat)) for pat in
                           ['*.png','*.nii.png','*.PNG','*.NII.PNG']], []))

        img_map = {_norm_stem(os.path.basename(p)): p for p in imgs}
        msk_map = {_norm_stem(os.path.basename(p)): p for p in msks}

        keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
        self.pairs: List[Tuple[str,str]] = [(img_map[k], msk_map[k]) for k in keys]

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No (image,mask) pairs found.\n  imgs={img_dir}\n  msks={msk_dir}\n"
                f"Examples of images: {list(img_map.values())[:5]}\n"
                f"Examples of masks: {list(msk_map.values())[:5]}"
            )

    def __len__(self): return len(self.pairs)

    def _load_gray(self, path: str) -> np.ndarray:
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
        return arr

    def _load_label(self, path: str) -> np.ndarray:
        # Masks are already color-mapped PNGs with small integer classes {0..C-1}
        arr = np.array(Image.open(path).convert('P'), dtype=np.int64)
        return arr

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        x = self._load_gray(img_path)
        y = self._load_label(msk_path)
        # center crop/resize to square img_size (optional; OASIS pngs are already square)
        if x.shape[0] != x.shape[1]:
            s = min(x.shape)
            sy = (x.shape[0]-s)//2; sx = (x.shape[1]-s)//2
            x = x[sy:sy+s, sx:sx+s]; y = y[sy:sy+s, sx:sx+s]
        if x.shape[0] != self.img_size:
            x = np.array(Image.fromarray((x*255).astype(np.uint8)).resize((self.img_size,self.img_size), Image.BILINEAR), dtype=np.float32)/255.0
            y = np.array(Image.fromarray(y.astype(np.uint8)).resize((self.img_size,self.img_size), Image.NEAREST), dtype=np.int64)
        x = torch.from_numpy(x)[None, ...]     # [1,H,W]
        y = torch.from_numpy(y)                # [H,W] (class indices 0..C-1)
        return x, y


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def dice_per_class_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    logits: [N,C,H,W], target: [N,H,W] (long, 0..C-1)
    returns: tensor [C] dice per class (ignore classes not present in union)
    """
    N, C, H, W = logits.shape
    pred = torch.argmax(logits, dim=1)  # [N,H,W]
    dices = []
    for c in range(C):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        dice = (2*inter + eps) / (union + eps)
        dices.append(dice)
    return torch.stack(dices, dim=0)  # [C]


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-test', required=True, help='folder of test images (PNG or .nii.png)')
    ap.add_argument('--msk-test', required=True, help='folder of test masks  (PNG or .nii.png)')
    ap.add_argument('--ckpt', required=True, help='path to checkpoint .pth')
    ap.add_argument('--classes', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=160)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--vis', action='store_true', help='save a few (img, pred, gt) tiles')
    ap.add_argument('--vis-out', default='results/test_infer', help='where to save visualizations')
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device.type)

    ds = OASISPNGDataset(args.img_test, args.msk_test, img_size=args.img_size)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = UNet(n_classes=args.classes).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[Warning] load_state_dict strict=False')
        if missing:   print('  missing:', missing)
        if unexpected:print('  unexpected:', unexpected)
    model.eval()

    all_dice = []
    os.makedirs(args.vis_out, exist_ok=True)
    for i, (x, y) in enumerate(ld, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)  # [N,C,H,W]
        d = dice_per_class_from_logits(logits, y)  # [C]
        all_dice.append(d.cpu())

        if args.vis and i <= 12:  # save first 12 mini-batches
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            img  = (x[:,0].cpu().numpy()*255).astype(np.uint8)
            gt   = y.cpu().numpy()
            # save triplets
            for k in range(min(len(img), 4)):
                out = np.zeros((args.img_size, args.img_size*3), dtype=np.uint8)
                out[:, :args.img_size] = img[k]
                out[:, args.img_size:args.img_size*2] = (pred[k] * 80).astype(np.uint8)
                out[:, args.img_size*2:] = (gt[k]   * 80).astype(np.uint8)
                Image.fromarray(out).save(os.path.join(args.vis_out, f'batch{i:03d}_{k}.png'))

        if i % 20 == 0:
            print(f'[{i}/{len(ld)}] running...')

    dices = torch.stack(all_dice, dim=0).mean(0)  # [C]
    print('\n==== Test set Dice (per class) ====')
    for c, v in enumerate(dices.tolist()):
        print(f'class {c}: {v:.4f}')
    print(f'Mean Dice: {dices.mean().item():.4f}')


if __name__ == '__main__':
    main()
