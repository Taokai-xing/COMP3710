import torch
import torch.nn.functional as F

@torch.no_grad()
def dice_per_class_from_logits(logits, target, eps=1e-6):
    """
    Compute per-class Dice from logits and integer mask.
    logits: [N, C, H, W] (raw scores)
    target: [N, H, W]    (long, values in [0..C-1])
    returns: tensor [C] dice for each class
    """
    N, C, H, W = logits.shape
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()  # [N,C,H,W]

    # intersection and union over N,H,W per class
    inter = (probs * onehot).sum(dim=(0,2,3))
    union = probs.sum(dim=(0,2,3)) + onehot.sum(dim=(0,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice  # [C]

class DiceLoss(torch.nn.Module):
    """
    Multi-class soft Dice loss (1 - mean Dice).
    Use with CrossEntropy for stable optimization: loss = CE + lambda * DiceLoss
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        dice = dice_per_class_from_logits(logits, target, eps=self.eps)
        return 1.0 - dice.mean()
