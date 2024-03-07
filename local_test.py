import os
import yaml
import json
import torch
import argparse
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")

from time import time
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils import parse_args_as_dict
from src.models import AVNet, videomodels
from src.system import System, make_optimizer
from src.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr


class AVSpeechDataset(Dataset):
    def __init__(self, epochs=5, audio_only=False):
        super().__init__()
        self.length = epochs
        self.audio_only = audio_only

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if not self.audio_only:
            return (torch.rand(32000), torch.rand(32000), torch.rand((1, 50, 88, 88)), "sample text")
        else:
            return (torch.rand(32000), torch.rand(32000), "sample text")


def build_dataloaders(conf, bs=None):
    bs = conf["training"]["batch_size"] if bs is None else bs
    audio_only = conf["data"].get("audio_only", False)

    train_set = AVSpeechDataset(500 * bs, audio_only)
    val_set = AVSpeechDataset(60 * bs, audio_only)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=bs, num_workers=conf["training"]["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=bs, num_workers=conf["training"]["num_workers"], drop_last=True)

    return train_loader, val_loader


def main(conf, epochs=1, bs=None):
    train_loader, val_loader = build_dataloaders(conf, bs)

    conf["videonet"] = conf.get("videonet", {})
    conf["videonet"]["model_name"] = conf["videonet"].get("model_name", None)

    # Define model and optimizer
    videomodel = None
    if conf["videonet"]["model_name"]:
        videomodel = videomodels.get(conf["videonet"]["model_name"])(**conf["videonet"])

    audiomodel = AVNet(**conf["audionet"])

    if conf["main_args"]["check_only"]:
        exit()

    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)

    # Just after instantiating, save the args. Easy loading in the future.
    conf["main_args"]["exp_dir"] = os.path.join("../experiments/audio-visual", "testing")
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yaml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = {
        "train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx"),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
    }

    # define system
    system = System(
        audio_model=audiomodel,
        video_model=videomodel,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=15, verbose=True))

    # default logger used by trainer
    comet_logger = TensorBoardLogger("./logs", name=conf["log"]["exp_name"])

    # instantiate ptl trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=[0],
        num_nodes=conf["main_args"]["nodes"],
        accelerator="auto",
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
    )

    trainer.fit(system)

    # Save best_k models
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # put on cpu and serialize
    state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
    system.load_state_dict(state_dict=state_dict["state_dict"])

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf-dir", type=str, default="config/lrs2_RTFSNet_4_layer.yaml", help="config path")
    parser.add_argument("-n", "--name", default=None, help="Experiment name")
    parser.add_argument("--nodes", type=int, default=1, help="#node")
    parser.add_argument("--check-only", type=bool, default=False, help="Only check params and MACs")

    args = parser.parse_args()
    cf_dir1 = str(args.conf_dir).split("/")[-1]

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf["log"]["exp_name"] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)

    t0 = time()
    main(def_conf)
    t1 = time()

    print("{}: {:.2f} seconds".format(cf_dir1, t1 - t0))
