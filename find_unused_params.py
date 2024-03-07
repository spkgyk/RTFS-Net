import os
import yaml
import torch
import argparse

torch.set_float32_matmul_precision("high")

from src.models import TDAVNet
from src.utils import parse_args_as_dict
from src.system import make_optimizer
from src.losses import PITLossWrapper, pairwise_neg_snr


x = torch.rand(2, 32000).to(0)
z = torch.rand(2, 1, 32000).to(0)
y = torch.rand(2, 512, 50).to(0)


def main(conf):
    audiomodel = TDAVNet(**conf["audionet"]).to(0)

    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx").to(0)
    optimizer.zero_grad()

    z1 = audiomodel(x, y)

    loss = loss_func(z1, z)
    loss.backward()

    for name, param in audiomodel.named_parameters():
        if param.grad is None:
            print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf-dir", default="config/lrs2_RTFSNet_4_layer.yaml")

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)
