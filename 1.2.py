
import os
os.environ["MPLBACKEND"] = "Agg"   # 无界面后端：保存图片
os.environ["OMP_NUM_THREADS"] = "1"
# 如仍遇到 OMP #15，可临时解围（跑通后建议删）：
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========= 导入 =========
import numpy as np
import torch
import matplotlib.pyplot as plt

# ========= 设备选择 =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device, "| CUDA available:", torch.cuda.is_available())

# ========= 颜色映射（与实验手册类似） =========
def colorize(ns_np: np.ndarray) -> np.ndarray:
    """把迭代计数矩阵映射成彩色图（uint8 HxWx3）"""
    a = ns_np.astype(np.float32)
    a_cyclic = (6.28 * a / 20.0)[..., np.newaxis]   # HxWx1
    img = np.concatenate([
        10  + 20*np.cos(a_cyclic),
        30  + 50*np.sin(a_cyclic),
        155 - 80*np.cos(a_cyclic)
    ], axis=2)
    img[a == a.max()] = 0                           # 内部（近似收敛）涂黑
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ========= 通用：生成复平面网格 =========
def complex_grid(xmin, xmax, ymin, ymax, step, device):
    # 与讲义一致：Y,X = np.mgrid[ymin:ymax:step, xmin:xmax:step]
    Y, X = np.mgrid[ymin:ymax:step, xmin:xmax:step]
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(Y, dtype=torch.float32, device=device)
    z = torch.complex(x, y)  # complex64
    return z

# ========= Mandelbrot =========
@torch.no_grad()
def mandelbrot(xmin=-2.0, xmax=1.0, ymin=-1.3, ymax=1.3, step=0.003, max_iter=200):
    """返回迭代计数 ns（越大表示越晚发散；最大值≈max_iter 的点近似收敛）"""
    c = complex_grid(xmin, xmax, ymin, ymax, step, device)
    z = torch.zeros_like(c)
    ns = torch.zeros(c.shape, dtype=torch.int32, device=device)

    # 迭代：z <- z^2 + c，并累计未发散的次数
    for _ in range(max_iter):
        z = z*z + c
        # 使用半径^2（避免开方），阈值 4
        not_diverged = (z.real*z.real + z.imag*z.imag) <= 4.0
        ns += not_diverged.to(torch.int32)

    return ns

# ========= Julia =========
@torch.no_grad()
def julia(c_const=(-0.8, 0.156),
          xmin=-1.7, xmax=1.7, ymin=-1.5, ymax=1.5, step=0.5, max_iter=200):
    """固定常数 c_const，初值为网格点 z0"""
    z = complex_grid(xmin, xmax, ymin, ymax, step, device)
    c = torch.complex(torch.tensor(c_const[0], device=device, dtype=torch.float32),
                      torch.tensor(c_const[1], device=device, dtype=torch.float32))
    ns = torch.zeros(z.shape, dtype=torch.int32, device=device)

    for _ in range(max_iter):
        z = z*z + c
        not_diverged = (z.real*z.real + z.imag*z.imag) <= 4.0
        ns += not_diverged.to(torch.int32)

    return ns

# ========= 保存图像工具 =========
def save_ns(ns_t: torch.Tensor, out_png: str, title: str):
    img = colorize(ns_t.detach().cpu().numpy())
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

# ========= 示例：Mandelbrot 基础图 =========
ns_m = mandelbrot(xmin=-2.0, xmax=1.0, ymin=-1.3, ymax=1.3, step=0.003, max_iter=200)
save_ns(ns_m, "mandelbrot.png", "Mandelbrot (step=0.003, iter=200)")

# ========= 示例：放大/高分辨率（用于演示 2.3 的“缩放+高分辨率”）=========
# 例：围绕著名的 Seahorse Valley 区域附近
ns_zoom = mandelbrot(xmin=-0.82, xmax=-0.68, ymin=-0.2, ymax=0.1, step=0.0015, max_iter=400)
save_ns(ns_zoom, "mandelbrot_zoom.png", "Mandelbrot Zoom (step=0.0015, iter=400)")

# ========= 示例：Julia 集（用于演示 2.3“改成 Julia”）=========
ns_j = julia(c_const=(-0.8, 0.156), xmin=-1.7, xmax=1.7, ymin=-1.5, ymax=1.5, step=0.003, max_iter=300)
save_ns(ns_j, "julia.png", "Julia c = -0.8 + 0.156i (iter=300)")
