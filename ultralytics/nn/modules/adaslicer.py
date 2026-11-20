import copy
import torch
import torch.nn as nn


class ObjSeeker(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=13, padding=6, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)
        self.proj = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        x = self.act(self.bn(self.dwconv(x)))
        return self.proj(x)


class AdaSlicerGate(nn.Module):
    """Shape-preserving soft objectness gate. c2 == c1 (channel count unchanged)."""

    def __init__(self, c1):
        super().__init__()
        self.seeker = ObjSeeker(c1)
        self.last_logits = None

    def forward(self, x):
        logits = self.seeker(x)
        self.last_logits = logits
        return x * torch.sigmoid(logits)

    def __deepcopy__(self, memo):
        """
        Excludes `last_logits` from deepcopy. It's a transient, forward-pass-only
        tensor that may still be attached to the autograd graph (non-leaf) when
        Ultralytics deepcopies the model to build the EMA shadow model -- PyTorch
        cannot deepcopy non-leaf tensors, which otherwise crashes model construction.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        saved = self.__dict__.pop("last_logits", None)
        try:
            new_obj.__dict__.update(copy.deepcopy(self.__dict__, memo))
        finally:
            self.__dict__["last_logits"] = saved  # restore on the original, untouched
        new_obj.last_logits = None  # the copy starts fresh — correct, since it's transient
        return new_obj