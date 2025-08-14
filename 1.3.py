# barnsley_torch.py
# 用 PyTorch 生成 Barnsley Fern（张量库版本，满足课程“用 TF/PyTorch”要求）

import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"   # 降低 OMP 冲突概率

import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def barnsley_fern_torch(points=200_000, H=1200, W=800, cmap="viridis", out="barnsley.png"):
    x = torch.tensor(0.0, device=device)
    y = torch.tensor(0.0, device=device)
    xs = torch.empty(points, device=device)
    ys = torch.empty(points, device=device)

    for i in range(points):
        r = torch.rand((), device=device)
        if r < 0.01:                      # f1
            x, y = 0.0 * x, 0.16 * y
        elif r < 0.86:                    # f2
            x, y = 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
        elif r < 0.93:                    # f3
            x, y = 0.20*x - 0.26*y, 0.23*x + 0.22*y + 1.6
        else:                             # f4
            x, y = -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
        xs[i], ys[i] = x, y

    xs = xs.detach().cpu().numpy()
    ys = ys.detach().cpu().numpy()

    xmin, xmax, ymin, ymax = -2.5, 2.5, 0.0, 10.0
    xi = np.clip(((xs - xmin) / (xmax - xmin) * (W - 1)).astype(np.int32), 0, W - 1)
    yi = np.clip(((ys - ymin) / (ymax - ymin) * (H - 1)).astype(np.int32), 0, H - 1)

    img = np.zeros((H, W), dtype=np.uint32)
    np.add.at(img, (yi, xi), 1)
    img = np.log1p(img.astype(np.float32))
    img /= img.max()

    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img, cmap=cmap, origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}  |  device={device}")

if __name__ == "__main__":
    barnsley_fern_torch()
