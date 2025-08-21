
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| device:", device)

@torch.inference_mode()
def barnsley_fern_parallel(
    batch=1_000_000,   
    steps=100,         
    H=1200, W=800,
    bounds=(-2.5, 2.5, 0.0, 10.0),  # xmin, xmax, ymin, ymax
    out="barnsley_parallel.png"
):
    xmin, xmax, ymin, ymax = bounds

   
    x = torch.zeros(batch, device=device)
    y = torch.zeros(batch, device=device)

   
    for _ in range(steps):
        r = torch.rand(batch, device=device)

        m1 = r < 0.01
        m2 = (r >= 0.01) & (r < 0.86)
        m3 = (r >= 0.86) & (r < 0.93)
        m4 = r >= 0.93

        # 备份旧坐标，避免就地修改覆盖问题
        x_old, y_old = x, y

        # f1: x'=0.00*x + 0.00*y + 0.00, y'=0.00*x + 0.16*y + 0.00
        # 注意：只更新对应掩码下的子集
        x = torch.where(m1, 0.0 * x_old, x_old)
        y = torch.where(m1, 0.16 * y_old, y_old)

        # f2: x'=0.85x + 0.04y, y'=-0.04x + 0.85y + 1.6
        x = torch.where(m2, 0.85 * x_old + 0.04 * y_old, x)
        y = torch.where(m2, -0.04 * x_old + 0.85 * y_old + 1.6, y)

        # f3: x'=0.20x - 0.26y, y'=0.23x + 0.22y + 1.6
        x = torch.where(m3, 0.20 * x_old - 0.26 * y_old, x)
        y = torch.where(m3, 0.23 * x_old + 0.22 * y_old + 1.6, y)

        # f4: x'=-0.15x + 0.28y, y'=0.26x + 0.24y + 0.44
        x = torch.where(m4, -0.15 * x_old + 0.28 * y_old, x)
        y = torch.where(m4, 0.26 * x_old + 0.24 * y_old + 0.44, y)

    # 3) 连续坐标 → 像素索引（全在 GPU）
    xi = ((x - xmin) / (xmax - xmin) * (W - 1)).clamp(0, W - 1).to(torch.int32)
    yi = ((y - ymin) / (ymax - ymin) * (H - 1)).clamp(0, H - 1).to(torch.int32)

    # 4) 2D 直方图：用 1D 索引 + torch.bincount（GPU 上统计）
    flat_idx = yi * W + xi
    hist = torch.bincount(flat_idx, minlength=H * W).reshape(H, W).to(torch.float32)

    # 对数拉伸 + 归一化（仍在 GPU）
    img = torch.log1p(hist)
    img = img / (img.max() + 1e-8)

    # 5) 拷回 CPU 出图
    img_np = img.cpu().numpy()
    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img_np, cmap="viridis", origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    barnsley_fern_parallel()

