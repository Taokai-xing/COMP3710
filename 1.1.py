# ========= 稳跑设置（必须放在最前面） =========
import os
os.environ["MPLBACKEND"] = "Agg"   # 无界面后端，保存图片，避免 Qt 弹窗
os.environ["OMP_NUM_THREADS"] = "1"  # 降低 OpenMP 冲突概率

# ========= 导入 =========
import numpy as np
import torch
import matplotlib.pyplot as plt

# ========= 设备选择 =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device, "| CUDA available:", torch.cuda.is_available())
print("Versions -> NumPy:", np.__version__, "| Torch:", torch.__version__)

# ========= 小工具：保存图 =========
def save_plot(tensor2d: torch.Tensor, filename: str, title: str = ""):
    arr = tensor2d.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    if title:
        plt.title(title)
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# ========= 网格（任务第4步） =========
# 范围与步长可调：步长越小分辨率越高但更耗时/显存
x_min, x_max, step = -4.0, 4.0, 0.01
X, Y = np.mgrid[x_min:x_max:step, x_min:x_max:step]  # 方形网格

# Numpy -> Torch，并放到设备上
x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)

# ========= 1) 二维高斯 =========
# 这里 σ=1：exp(-(x^2+y^2)/2)
z_gauss = torch.exp(-(x**2 + y**2) / 2.0)
save_plot(z_gauss, "gaussian.png", "2D Gaussian")

# ========= 2) 二维正弦/余弦条纹 =========
# 条纹频率与方向可调：
# 频率 freq 表示“每单位长度的周期数（cycles/unit）”
# 方向 theta（弧度）：0 沿 +x 方向；pi/2 沿 +y 方向
freq = 0.5         # 调大：更密的条纹；调小：更疏
theta = np.deg2rad(30)  # 条纹方向（这里示例 30 度）
fx = freq * np.cos(theta)
fy = freq * np.sin(theta)

two_pi = 2.0 * np.pi
phase = 0.0  # 相位，可改成任意浮点数
z_sin = torch.sin(two_pi * (fx * x + fy * y) + phase)
z_cos = torch.cos(two_pi * (fx * x + fy * y) + phase)

save_plot(z_sin, "sine2d.png", "2D Sine stripes")
save_plot(z_cos, "cosine2d.png", "2D Cosine stripes")

# ========= 3) 调制（Gabor：高斯 × 正弦/余弦） =========
z_gabor_sin = z_gauss * z_sin
z_gabor_cos = z_gauss * z_cos

save_plot(z_gabor_sin, "gabor_sin.png", "Gabor (Gaussian × Sine)")
save_plot(z_gabor_cos, "gabor_cos.png", "Gabor (Gaussian × Cosine)")

print("All done. Files: gaussian.png, sine2d.png, cosine2d.png, gabor_sin.png, gabor_cos.png")
