import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_skeletonize(img, iters=10):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_erode(img))
    for _ in range(iters):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_erode(img))
    return torch.clamp(skel, 0.0, 1.0)


class SoftclDiceLoss(nn.Module):
    """Differentiable Centerline Dice Loss (Shit et al., CVPR 2021)."""
    def __init__(self, iters=10, smooth=1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        skel_pred = soft_skeletonize(y_pred, self.iters)
        skel_true = soft_skeletonize(y_true, self.iters)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(y_pred * skel_true) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)
        return 1.0 - cldice


def soft_dice(prob, target, smooth=1.0):
    num = 2 * (prob * target).sum() + smooth
    den = prob.sum() + target.sum() + smooth
    return 1.0 - num / den


class CombinedLoss(nn.Module):
    """L = BCE + Dice + w_cldice*clDice on foreground, plus an auxiliary
    (BCE + Dice) on the spine head. Weights are starting points -- tune on the
    validation PQ, not on the loss value."""
    def __init__(self, w_cldice=0.5, w_spine=0.5, cldice_iters=10):
        super().__init__()
        self.w_cldice = w_cldice
        self.w_spine = w_spine
        self.cldice = SoftclDiceLoss(iters=cldice_iters)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, out, target):
        seg_logit, sp_logit = out["seg"], out["spine"]
        fg, sp = target["fg"], target["spine"]
        seg_prob = torch.sigmoid(seg_logit)
        sp_prob = torch.sigmoid(sp_logit)

        l_seg = self.bce(seg_logit, fg) + soft_dice(seg_prob, fg) + self.w_cldice * self.cldice(seg_prob, fg)
        l_spine = self.bce(sp_logit, sp) + soft_dice(sp_prob, sp)
        total = l_seg + self.w_spine * l_spine
        return total, {"seg": float(l_seg), "spine": float(l_spine)}
