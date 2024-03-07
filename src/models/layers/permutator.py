import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))


def ReturnPermutator(*args, image_size, in_chan, patch_size, dim, depth, segments, expansion_factor=4, dropout=0.0, **kwargs):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, "image must be divisible by patch size"
    assert (dim % segments) == 0, "dimension must be divisible by the number of segments"
    height, width = image_h // patch_size, image_w // patch_size
    s = segments

    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_size, p2=patch_size),
        nn.Linear((patch_size**2) * in_chan, dim),
        *[
            nn.Sequential(
                PreNormResidual(
                    dim,
                    nn.Sequential(
                        ParallelSum(
                            nn.Sequential(
                                Rearrange("b h w (c s) -> b w c (h s)", s=s),
                                nn.Linear(height * s, height * s),
                                Rearrange("b w c (h s) -> b h w (c s)", s=s),
                            ),
                            nn.Sequential(
                                Rearrange("b h w (c s) -> b h c (w s)", s=s),
                                nn.Linear(width * s, width * s),
                                Rearrange("b h c (w s) -> b h w (c s)", s=s),
                            ),
                            nn.Linear(dim, dim),
                        ),
                        nn.Linear(dim, dim),
                    ),
                ),
                PreNormResidual(
                    dim,
                    nn.Sequential(
                        nn.Linear(dim, dim * expansion_factor),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * expansion_factor, dim),
                        nn.Dropout(dropout),
                    ),
                ),
            )
            for _ in range(depth)
        ],
        nn.LayerNorm(dim),
        nn.Linear(dim, (patch_size**2) * in_chan),
        Rearrange("b h w (p1 p2 c) -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, h=height, w=width),
    )


class Permutator(nn.Module):
    def __init__(self, patch_size, image_size, *args, **kwargs):
        super(Permutator, self).__init__()

        image_size = tuple(image_size)
        self.patch_size = patch_size
        self.mlp = ReturnPermutator(*args, patch_size=patch_size, image_size=image_size, **kwargs)
        self.first_it = True

    def forward(self, x: torch.Tensor):
        old_w, old_h = x.shape[-2:]
        new_w = (old_w // self.patch_size) * self.patch_size + self.patch_size - old_w
        new_h = (old_h // self.patch_size) * self.patch_size + self.patch_size - old_h
        x = nn.functional.pad(x, (0, new_h, 0, new_w))

        if self.first_it:
            print("Old Shape: [{},{}]".format(old_w, old_h))
            print("New shape: [{},{}]".format(new_w + old_w, new_h + old_h))
            self.first_it = False

        x = self.mlp(x)
        x = x[..., :old_w, :old_h]
        return x
