import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(dense(dim, inner_dim), nn.GELU(), nn.Dropout(dropout), dense(inner_dim, dim), nn.Dropout(dropout))


def MLPMixer(*args, image_size, in_chan, patch_size, dim, depth, expansion_factor=4, expansion_factor_token=0.5, dropout=0.0, **kwargs):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, "image must be divisible by patch size"
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
        nn.Linear((patch_size**2) * in_chan, dim),
        *[
            nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last)),
            )
            for _ in range(depth)
        ],
        nn.LayerNorm(dim),
        nn.Linear(dim, (patch_size**2) * in_chan),
        Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=image_h // patch_size, w=image_w // patch_size, p1=patch_size, p2=patch_size),
    )


class MLP(nn.Module):
    def __init__(self, patch_size, image_size, *args, **kwargs):
        super(MLP, self).__init__()

        image_size = tuple(image_size)
        self.patch_size = patch_size
        self.mlp = MLPMixer(*args, patch_size=patch_size, image_size=image_size, **kwargs)
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
