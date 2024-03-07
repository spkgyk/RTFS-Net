import os
import json
import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models.videomodels.autoencoder.datamodule import AVSpeechDataModule
from src.models.videomodels.autoencoder.autoencoder import AE

ckpt_name = None


def main():
    # dataloader
    datamodule = AVSpeechDataModule(
        "data-preprocess/LRS2/tr",
        "data-preprocess/LRS2/cv",
        "data-preprocess/LRS2/tt",
        segment=2,
        batch_size=40,
    )
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader
    # Define scheduler
    system = AE(in_channels=1, base_channels=4, num_layers=3, train_loader=train_loader, val_loader=val_loader)

    # Define callbacks
    callbacks = []
    exp_dir = os.path.join("../experiments/autoencoder", "default")
    checkpoint = ModelCheckpoint(
        exp_dir,
        filename="{epoch}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    callbacks.append(EarlyStopping(monitor="val/loss", patience=10, verbose=True))

    # default logger used by trainer
    comet_logger = TensorBoardLogger(exp_dir, name="baseline")

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices="auto",
        accelerator="auto",
        strategy=DDPStrategy(),
        limit_train_batches=1.0,
        logger=comet_logger,
    )

    trainer.fit(system, ckpt_path=os.path.join(exp_dir, ckpt_name) if ckpt_name else None)
    print("Finished Training")

    # Save best_k models
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # put on cpu and serialize
    state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
    system.load_state_dict(state_dict=state_dict["state_dict"])

    to_save = system.encoder.state_dict()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
