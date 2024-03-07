import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        leaky_slope=0.3,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        leaky_slope=0.3,
    ) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EncoderAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=8,
        num_layers=3,
    ) -> None:
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            cout = base_channels * (2**i)
            cin = in_channels if i == 0 else cout // 2
            self.layers.append(EncoderBlock(cin, cout, 2, 2))
        self.layers = nn.Sequential(*self.layers)
        self.out_channels = cout

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            # print("[E] x", x.shape)
        return x


class DecoderAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=8,
        num_layers=3,
    ) -> None:
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            cin = base_channels * (2 ** (num_layers - i - 1))
            cout = in_channels if i == num_layers - 1 else cin // 2
            self.layers.append(DecoderBlock(cin, cout, 2, 2))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            # print("[D] x", x.shape)
        return x


class AE(LightningModule):
    def __init__(
        self,
        in_channels=1,
        base_channels=8,
        num_layers=3,
        # Training parameters
        criterion=nn.MSELoss(),
        train_loader=None,
        val_loader=None,
    ) -> None:
        super().__init__()
        self.train_loader = (train_loader,)
        self.val_loader = (val_loader,)
        self.default_monitor = "val/loss"
        # Access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.encoder = EncoderAE(in_channels, base_channels, num_layers)
        self.decoder = DecoderAE(in_channels, base_channels, num_layers)

        self.criterion = criterion

        self.optimizer = torch.optim.Adam(
            params=[{"params": self.encoder.parameters()}, {"params": self.decoder.parameters()}],
            lr=1e-4,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, factor=0.5)

    def forward(self, x):
        # x is expected to be a tensor of shape [batch, frames, w, h].
        # Convert it to [batch * frames, w, h]
        batch, frames, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.contiguous().view(batch * frames, 1, w, h)

        z = self.encoder(x)

        # Undo the view of x. z has [batch * frames, c', w', h']
        # Convert it to [batch, frames, c' * w' * h']
        z = z.view(batch, frames, -1)
        return z

    def reconstruct(self, x):
        # x is expected to be a tensor of shape [batch, frames, w, h].
        # Convert it to [batch * frames, w, h]
        batch, frames, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(batch * frames, 1, w, h)

        z = self.encoder(x)
        y = self.decoder(z)

        # Undo the view of x. y has [batch * frames, w, h]
        # Convert it to [batch, frames, w, h]
        y = y.view(batch, frames, w, h)
        return y

    def step(self, batch):
        x = batch
        preds = self.reconstruct(x)
        loss = self.criterion(preds, x)
        return loss, preds

    def training_step(self, batch, batch_idx: int):
        loss, preds = self.step(batch[0])
        # Log train metrics
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds}

    def validation_step(self, batch, batch_idx: int):
        loss, preds = self.step(batch)
        # Log train metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": preds}

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers


if __name__ == "__main__":
    model = AE(in_channels=1, base_channels=4, num_layers=3)

    batch = 2
    x = torch.randn(batch, 50, 88, 88)

    y = model(x)
    print("y", y.shape)

    z = model.reconstruct(x)
    print("z", z.shape)
