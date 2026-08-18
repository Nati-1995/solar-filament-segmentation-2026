"""Compact U-Net with a shared encoder and two heads: filament foreground and
spine/centerline. The spine head both supervises structure (via clDice) and
seeds the watershed at inference. This is the reliable baseline from the review
-- robust on the small MAGFiLO set. Mask2Former is the stretch upgrade and can
reuse the same preprocessing, dataset, loss, and PQ code.
"""
import torch
import torch.nn as nn


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNetSpine(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.d1 = double_conv(in_ch, c[0])
        self.d2 = double_conv(c[0], c[1])
        self.d3 = double_conv(c[1], c[2])
        self.bott = double_conv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.u3 = double_conv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.u2 = double_conv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u1 = double_conv(c[1], c[0])
        self.seg_head = nn.Conv2d(c[0], 1, 1)
        self.spine_head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        xb = self.bott(self.pool(x3))
        y = self.u3(torch.cat([self.up3(xb), x3], 1))
        y = self.u2(torch.cat([self.up2(y), x2], 1))
        y = self.u1(torch.cat([self.up1(y), x1], 1))
        return {"seg": self.seg_head(y), "spine": self.spine_head(y)}
