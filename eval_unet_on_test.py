# -*- coding: utf-8 -*-
#cd "C:\Users\21590\1.4\unet"
#python .\eval_unet_on_test.py --save-pred --save-color --save-overlay --workers 0

import os, re, sys, argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

CASE_RE = re.compile(r"(?:^|_)case_(\d+)_slice_(\d+)", re.IGNORECASE)
SEG_RE  = re.compile(r"(?:^|_)seg_(\d+)_slice_(\d+)",  re.IGNORECASE)

def norm_key_from_name(name: str):
    base = Path(name).name.lower()
    base = base.replace(".nii.png", "").replace(".png", "").replace(".nii", "")
    m = CASE_RE.search(base) or SEG_RE.search(base)
    if not m: return None
    pid, sl = m.group(1), m.group(2)
    return f"{int(pid):03d}_slice_{int(sl)}"

def clean_stem(p: Path):
    s = p.name
    for suf in (".nii.png", ".png", ".nii", ".jpg", ".jpeg"):
        if s.endswith(suf): return s[:-len(suf)]
    return p.stem

def imread_gray(path: Path):
    return Image.open(str(path)).convert("L")

def to_tensor01(img_pil: Image.Image):
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

def resize_pair(x_pil: Image.Image, y_pil: Image.Image, size: int):
    x = x_pil.resize((size, size), Image.BILINEAR)
    y = y_pil.resize((size, size), Image.NEAREST)
    return x, y

def mask_to_index(y_pil: Image.Image, classes: int):
    y = np.array(y_pil, dtype=np.float32)
    if y.ndim == 3: y = y.mean(axis=2)
    y = np.round((y / 255.0) * (classes - 1)).astype(np.int64)
    y = np.clip(y, 0, classes - 1)
    return torch.from_numpy(y)  # [H,W], long

# =========================
# 数据集
# =========================
class OASISPNGPaired(Dataset):
    def __init__(self, img_dir, msk_dir, img_size=256, classes=4, raise_on_missing=True, return_names=False):
        self.img_dir = Path(img_dir); self.msk_dir = Path(msk_dir)
        self.img_size = img_size; self.classes = classes; self.return_names = return_names
        assert self.img_dir.exists(), f"Images dir not found: {self.img_dir}"
        assert self.msk_dir.exists(), f"Masks dir not found: {self.msk_dir}"

        img_files = [p for p in self.img_dir.iterdir() if p.suffix.lower().endswith("png")]
        msk_files = [p for p in self.msk_dir.iterdir() if p.suffix.lower().endswith("png")]

        key_to_mask = {}
        for p in msk_files:
            k = norm_key_from_name(p.name)
            if k: key_to_mask[k] = p

        pairs = []
        for p in img_files:
            k = norm_key_from_name(p.name)
            if not k: continue
            m = key_to_mask.get(k, None)
            if m is not None and m.exists():
                pairs.append((p, m))

        if len(pairs) == 0:
            examples = [p.name for p in img_files[:10]]
            raise FileNotFoundError(
                f"No (image,mask) pairs found.\n  imgs={self.img_dir}\n  msks={self.msk_dir}\n"
                f"Examples: {examples}"
            )

        if raise_on_missing:
            missing = len(img_files) - len(pairs)
            if missing > 0:
                print(f"[INFO] {missing} image(s) had no matching mask and were skipped.")
        self.pairs = pairs
        print(f"[OK] Found {len(self.pairs)} pairs")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        x_pil = imread_gray(img_path)
        y_pil = Image.open(str(msk_path))

        x_pil, y_pil = resize_pair(x_pil, y_pil, self.img_size)
        x = to_tensor01(x_pil)                        # [1,H,W]
        y = mask_to_index(y_pil, self.classes).long() # [H,W]
        if self.return_names:
            return x, y, clean_stem(img_path)
        return x, y

# =========================
# 内置 UNet（正确的 OutConv：Conv2d(in_ch=base, out_ch=n_classes, 1)）
# =========================
def get_builtin_unet_class():
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
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
        def forward(self, x, skip):
            x = self.up(x)
            dh = skip.size(2) - x.size(2); dw = skip.size(3) - x.size(3)
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
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
            self.inc   = DoubleConv(n_channels, base)
            self.down1 = Down(base, base*2)
            self.down2 = Down(base*2, base*4)
            self.down3 = Down(base*4, base*8)
            self.down4 = Down(base*8, base*16)
            self.up1 = Up(base*16, base*8,  base*8)
            self.up2 = Up(base*8,  base*4,  base*4)
            self.up3 = Up(base*4,  base*2,  base*2)
            self.up4 = Up(base*2,  base,    base)
            self.outc = OutConv(base, n_classes)
        def forward(self, x):
            x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
            x  = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
            return self.outc(x)
    return UNet

# =========================
# 导入你训练时的 UNet（优先同级、model/、models/）
# =========================
def import_unet_candidate(script_dir: Path):
    # 同级
    if (script_dir / "unet.py").exists():
        sys.path.insert(0, str(script_dir))
        try:
            from unet import UNet
            print(f"[OK] Imported UNet from {script_dir/'unet.py'}")
            return UNet
        except Exception as e:
            print(f"[WARN] Failed to import UNet from {script_dir/'unet.py'}: {e}")
    # model/
    model_dir = script_dir / "model"
    if (model_dir / "unet.py").exists():
        sys.path.insert(0, str(model_dir))
        try:
            from unet import UNet
            print(f"[OK] Imported UNet from {model_dir/'unet.py'}")
            return UNet
        except Exception as e:
            print(f"[WARN] Failed to import UNet from {model_dir}: {e}")
    # models/
    models_dir = script_dir / "models"
    if (models_dir / "unet.py").exists():
        sys.path.insert(0, str(models_dir))
        try:
            from unet import UNet
            print(f"[OK] Imported UNet from {models_dir/'unet.py'}")
            return UNet
        except Exception as e:
            print(f"[WARN] Failed to import UNet from {models_dir}: {e}")
    # 包方式
    sys.path.insert(0, str(script_dir))
    for pkg in ("model.unet", "models.unet"):
        try:
            mod = __import__(pkg, fromlist=["UNet"])
            UNet = getattr(mod, "UNet")
            print(f"[OK] Imported UNet via package '{pkg}'")
            return UNet
        except Exception as e:
            print(f"[WARN] Failed to import UNet via '{pkg}': {e}")
    return None  # 交给内置

# =========================
# Dice（累计 inter/denom）
# =========================
@torch.no_grad()
def dice_stats_from_logits(logits: torch.Tensor, target: torch.Tensor):
    N, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()
    inter = (probs * onehot).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + onehot.sum(dim=(0,2,3))
    return inter, denom

# =========================
# 可视化保存
# =========================
def default_palette(n_classes: int):
    colors = [
        (0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0),
        (255,0,255), (0,255,255), (255,128,0), (128,0,255), (0,128,255)
    ]
    if n_classes <= len(colors): return colors[:n_classes]
    import colorsys
    ext = []
    for i in range(n_classes - len(colors)):
        h = (i / max(1, n_classes-1))
        r,g,b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        ext.append((int(r*255), int(g*255), int(b*255)))
    return colors + ext

def save_index_png(path: Path, pred_idx: np.ndarray):
    Image.fromarray(pred_idx.astype(np.uint8), mode="L").save(str(path))

def save_color_png(path: Path, pred_idx: np.ndarray, palette_rgb):
    pal = []
    for (r,g,b) in palette_rgb: pal.extend([r,g,b])
    pal += [0,0,0] * (256 - len(palette_rgb))
    im = Image.fromarray(pred_idx.astype(np.uint8), mode="P")
    im.putpalette(pal)
    im.save(str(path))

def save_overlay_png(path: Path, img_gray_uint8: np.ndarray, pred_idx: np.ndarray, palette_rgb, alpha=110):
    H,W = img_gray_uint8.shape
    base = Image.fromarray(img_gray_uint8, mode="L").convert("RGBA")
    overlay = np.zeros((H,W,4), dtype=np.uint8)
    for cls,(r,g,b) in enumerate(palette_rgb):
        if cls == 0: continue
        m = (pred_idx == cls)
        overlay[m,0]=r; overlay[m,1]=g; overlay[m,2]=b; overlay[m,3]=alpha
    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
    out.save(str(path))

# =========================
# Checkpoint & 维度推断
# =========================
def resolve_ckpt_path_bulletproof(arg_path: str, script_dir: Path) -> Path:
    raw = (arg_path or "").strip().strip('"').strip("'")
    candidates = []
    p = Path(raw)
    candidates.append(p if p.is_absolute() else (script_dir / p))
    candidates.append(script_dir / "checkpoints" / (p.name if p.name else "unet_oasis_png.pth"))
    candidates.append(script_dir.parent / "checkpoints" / (p.name if p.name else "unet_oasis_png.pth"))
    def swap_suffix(path: Path):
        if path.suffix.lower()==".pth": return path.with_suffix(".pt")
        if path.suffix.lower()==".pt":  return path.with_suffix(".pth")
        return None
    extra = []
    for c in candidates:
        alt = swap_suffix(c)
        if alt: extra.append(alt)
    candidates.extend(extra)
    # 递归搜
    for root in [script_dir, script_dir / "checkpoints", script_dir.parent / "checkpoints"]:
        if root.exists():
            candidates += list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    seen=set()
    for c in candidates:
        try: c=c.resolve()
        except: c=Path(str(c))
        if str(c) in seen: continue
        seen.add(str(c))
        if c.exists() and c.is_file(): return c
    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates[:12])
    cwd_info = f"\nCWD: {Path.cwd()}"
    raise FileNotFoundError(f"Checkpoint not found. Tried:{tried}\nScript dir: {script_dir}{cwd_info}")

def extract_state_dict(obj):
    if not isinstance(obj, dict):
        sd = obj
    else:
        for k in ["model","state_dict","model_state_dict","unet","net"]:
            if k in obj:
                obj = obj[k]; break
        sd = obj
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k,v in sd.items())

def infer_unet_dims_from_state(sd: OrderedDict):
    # 第一层
    cand_first = [k for k in sd if k.endswith(".weight") and "conv" in k and ("inc" in k or "down" in k)]
    if not cand_first: cand_first = [k for k in sd if k.endswith(".weight") and "conv" in k]
    cand_first.sort()
    if not cand_first: raise RuntimeError("Cannot locate first conv.")
    w0 = sd[cand_first[0]]          # [out, in, k, k]
    base = int(w0.shape[0]); in_ch = int(w0.shape[1])
    # 最后一层（优先 outc/out 的 1x1）
    cand_last = [k for k in sd if k.endswith(".weight") and ("outc" in k or ".out" in k or k.split(".")[-2]=="out")]
    if not cand_last:
        cand_1x1 = [k for k,v in sd.items() if k.endswith(".weight") and v.ndim==4 and v.shape[2]==1 and v.shape[3]==1]
        cand_last = cand_1x1 if cand_1x1 else [cand_first[-1]]
    cand_last.sort()
    wL = sd[cand_last[-1]]          # [n_classes, base, 1, 1]
    n_classes = int(wL.shape[0])
    return in_ch, n_classes, base

# =========================
# 参数
# =========================
def build_parser_with_defaults(script_dir: Path):
    ap = argparse.ArgumentParser(description="Evaluate UNet on OASIS PNG test set and export predictions.")
    ap.add_argument("--img-test", default="D:/3710/keras_png_slices_data/keras_png_slices_test", type=str)
    ap.add_argument("--msk-test", default="D:/3710/keras_png_slices_data/keras_png_slices_seg_test", type=str)
    ap.add_argument("--ckpt",     default=str(script_dir / "checkpoints" / "unet_oasis_png.pth"), type=str)
    ap.add_argument("--classes",  default=4, type=int)
    ap.add_argument("--img-size", default=256, type=int)
    ap.add_argument("--batch",    default=8, type=int)
    ap.add_argument("--workers",  default=0, type=int)
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--save-color", action="store_true")
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--outdir", default=str(script_dir / "results" / "test_infer"), type=str)
    ap.add_argument("--overlay-alpha", default=110, type=int)
    return ap

# =========================
# 主流程
# =========================
@torch.no_grad()
def main():
    script_dir = Path(__file__).resolve().parent
    parser = build_parser_with_defaults(script_dir)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 载入 ckpt 并推断维度
    ckpt_path = resolve_ckpt_path_bulletproof(args.ckpt, script_dir)
    print(f"[OK] Using checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = extract_state_dict(ckpt)
    try:
        in_ch_ckpt, n_classes_ckpt, base_ckpt = infer_unet_dims_from_state(state)
        if args.classes != n_classes_ckpt:
            print(f"[INFO] Override classes {args.classes} -> {n_classes_ckpt} (from ckpt)")
            args.classes = n_classes_ckpt
    except Exception as e:
        print(f"[WARN] Cannot infer dims from ckpt: {e}; fallback to args.")
        in_ch_ckpt, n_classes_ckpt, base_ckpt = 1, args.classes, 64

    # 数据
    ds = OASISPNGPaired(args.img_test, args.msk_test, img_size=args.img_size, classes=args.classes, raise_on_missing=False, return_names=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # 模型类候选：先尝试你自己的（model/等），若 outc 方向和 ckpt 不一致就退回内置
    UNet_builtin = get_builtin_unet_class()
    UNet_cand = import_unet_candidate(script_dir)

    def try_build_and_load(UNetClass, label):
        ctor_trials = [
            (in_ch_ckpt, args.classes, base_ckpt),
            (in_ch_ckpt, args.classes),
            (args.classes,),
            tuple(),
        ]
        err_log = []
        for t in ctor_trials:
            try:
                m = UNetClass(*t).to(device)
                # 如果有 outc.conv，检查方向是否和 ckpt 相符（应为 [classes, base, 1,1]）
                try:
                    w = m.outc.conv.weight
                    if tuple(w.shape[-2:]) == (1,1):
                        expect = (args.classes, base_ckpt, 1, 1)
                        rev    = (base_ckpt, args.classes, 1, 1)
                        if tuple(w.shape) == rev:
                            raise RuntimeError(f"Detected reversed OutConv shape {w.shape} vs expected {expect}")
                except Exception as _:
                    pass
                # 严格加载
                m.load_state_dict(state, strict=True)
                print(f"[OK] Loaded with {label} UNet and ctor args {t} (strict=True).")
                return m
            except Exception as e:
                err_log.append((t, str(e)))
        print(f"[INFO] Failed strict load with {label} UNet; details:")
        for a, e in err_log:
            print("  -", a, "=>", e[:300], "..." if len(e) > 300 else "")
        return None

    model = None
    # 先试候选（你的训练版），若 outc 方向不符会在上面抛异常从而失败
    if UNet_cand is not None:
        model = try_build_and_load(UNet_cand, "candidate")

    # 若失败，改用内置正确实现
    if model is None:
        model = try_build_and_load(UNet_builtin, "builtin")

    # 如果还不行，最后宽松加载
    if model is None:
        print("[WARN] Fallback: builtin UNet with strict=False.")
        model = UNet_builtin(in_ch_ckpt, args.classes, base_ckpt).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:   print("  missing keys:", missing)
        if unexpected:print("  unexpected keys:", unexpected)

    model.eval()

    # 输出目录
    outdir = Path(args.outdir)
    out_pred, out_color, out_overlay = outdir/"pred_idx", outdir/"pred_color", outdir/"overlay"
    if args.save_pred: ensure_dir(out_pred)
    if args.save_color: ensure_dir(out_color)
    if args.save_overlay: ensure_dir(out_overlay)
    ensure_dir(outdir)
    palette_rgb = default_palette(args.classes)

    # 累计 Dice
    inter_sum = torch.zeros(args.classes, device=device)
    denom_sum = torch.zeros(args.classes, device=device)

    for it, (x, y, names) in enumerate(dl, 1):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        inter, denom = dice_stats_from_logits(logits, y)
        inter_sum += inter; denom_sum += denom

        if args.save_pred or args.save_color or args.save_overlay:
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            imgs = (x[:,0].cpu().numpy()*255).astype(np.uint8)
            for b in range(pred.shape[0]):
                stem = str(names[b])
                if args.save_pred:
                    save_index_png(out_pred / f"{stem}_pred.png", pred[b])
                if args.save_color:
                    save_color_png(out_color / f"{stem}_pred_color.png", pred[b], palette_rgb)
                if args.save_overlay:
                    save_overlay_png(out_overlay / f"{stem}_overlay.png", imgs[b], pred[b], palette_rgb, alpha=int(args.overlay_alpha))

        if it % 20 == 0:
            print(f"[{it}/{len(dl)}] processed")

    eps = 1e-6
    per_class = ((2*inter_sum + eps) / (denom_sum + eps)).detach().cpu().numpy()
    mean_all = float(per_class.mean())
    mean_fg  = float(per_class[1:].mean()) if args.classes > 1 else mean_all

    print("\n==== Test Dice (per class) ====")
    for c, v in enumerate(per_class):
        print(f"class {c}: {v:.4f}")
    print(f"Mean Dice: {mean_all:.4f}")
    if args.classes > 1:
        print(f"Mean Dice (foreground only): {mean_fg:.4f}")

if __name__ == "__main__":
    main()


