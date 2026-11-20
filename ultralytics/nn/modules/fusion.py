import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ("ASFF", "ASFF2", "ASFF3", "HFP", "SDP")


class _ASFFBase(nn.Module):
    """Adaptive spatial fusion of N levels onto the resolution/channels of `level`."""

    def __init__(self, chs, level, weight_ch=16):
        super().__init__()
        assert 0 <= level < len(chs)
        self.level, self.n = level, len(chs)
        c = chs[level]

        self.resize = nn.ModuleList()
        for i, ci in enumerate(chs):
            if i == level:
                self.resize.append(nn.Identity())
            elif i < level:  # finer -> downsample by 2**(level-i)
                layers = [Conv(ci, c, 3, 2)] + [nn.MaxPool2d(3, 2, 1)] * (level - i - 1)
                self.resize.append(nn.Sequential(*layers))
            else:  # coarser -> 1x1 conv, upsample in forward
                self.resize.append(Conv(ci, c, 1, 1))

        self.weight_convs = nn.ModuleList(Conv(c, weight_ch, 1, 1) for _ in chs)
        self.weight_levels = nn.Conv2d(weight_ch * self.n, self.n, 1)
        self.out_conv = Conv(c, c, 3, 1)

    def forward(self, xs):
        size = xs[self.level].shape[-2:]
        feats = []
        for x, r in zip(xs, self.resize):
            x = r(x)
            if x.shape[-2:] != size:
                x = F.interpolate(x, size=size, mode="nearest")
            feats.append(x)
        w = torch.cat([wc(f) for wc, f in zip(self.weight_convs, feats)], 1)
        w = F.softmax(self.weight_levels(w), dim=1)
        return self.out_conv(sum(feats[i] * w[:, i:i + 1] for i in range(self.n)))


class ASFF3(_ASFFBase):
    def __init__(self, chs, level):
        assert len(chs) == 3
        super().__init__(chs, level)


class ASFF2(_ASFFBase):
    def __init__(self, chs, level):
        assert len(chs) == 2
        super().__init__(chs, level)


class ASFF(ASFF3):
    pass


class HFP(nn.Module):
    """Learnable-cutoff FFT high-pass + channel/spatial attention, residual."""

    def __init__(self, c, reduction=16, init_cutoff=0.1, temp=0.03):
        super().__init__()
        self.raw_cutoff = nn.Parameter(torch.tensor(math.log(init_cutoff / (0.5 - init_cutoff))))
        self.temp = temp
        hidden = max(c // reduction, 8)
        self.ca = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), nn.SiLU(), nn.Conv2d(hidden, c, 1, bias=False))
        self.sa = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _high_pass(self, x):
        B, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):  # FFT in fp32
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            fy = torch.fft.fftfreq(H, device=x.device).view(H, 1)
            fx = torch.fft.rfftfreq(W, device=x.device).view(1, W // 2 + 1)
            radius = torch.sqrt(fy ** 2 + fx ** 2)
            cutoff = 0.5 * torch.sigmoid(self.raw_cutoff.float())
            mask = torch.sigmoid((radius - cutoff) / self.temp)
            hf = torch.fft.irfft2(xf * mask, s=(H, W), norm="ortho")
        return hf.to(x.dtype)

    def forward(self, x):
        hf = self._high_pass(x)
        ca = torch.sigmoid(self.ca(F.adaptive_avg_pool2d(hf, 1)) + self.ca(F.adaptive_max_pool2d(hf, 1)))
        sa = torch.sigmoid(self.sa(torch.cat([hf.mean(1, keepdim=True), hf.amax(1, keepdim=True)], 1)))
        return x + self.gamma * (hf * ca * sa)


class SDP(nn.Module):
    """Cross-attention: query = fine map, key/value = coarse map. Output has the fine map's shape."""

    def __init__(self, chs, heads=4):
        super().__init__()
        c_f, c_c = chs
        while c_f % heads:
            heads -= 1
        self.heads = heads
        self.q = nn.Conv2d(c_f, c_f, 1, bias=False)
        self.kv = nn.Conv2d(c_c, 2 * c_f, 1, bias=False)
        self.proj = nn.Conv2d(c_f, c_f, 1, bias=False)
        self.norm_f, self.norm_c = nn.BatchNorm2d(c_f), nn.BatchNorm2d(c_c)
        self.refine = nn.Sequential(nn.Conv2d(c_f, c_f, 3, 1, 1, groups=c_f, bias=False), nn.BatchNorm2d(c_f), nn.SiLU())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, xs):
        fine, coarse = xs
        B, C, H, W = fine.shape
        h = self.heads
        q = self.q(self.norm_f(fine)).flatten(2).transpose(1, 2)
        k, v = self.kv(self.norm_c(coarse)).flatten(2).transpose(1, 2).chunk(2, dim=-1)
        q, k, v = (t.reshape(B, -1, h, C // h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)
        return self.refine(fine + self.gamma * self.proj(a))