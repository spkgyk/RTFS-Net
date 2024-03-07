###
# Author: Kai Li
# Date: 2021-06-18 17:29:21
# LastEditors: Kai Li
# LastEditTime: 2021-06-21 23:52:52
###

import torch
import subprocess
import torch.nn as nn


def pad_x_to_y(x, y, axis: int = -1):
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def shape_reconstructed(reconstructed, size):
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def tensors_to_device(tensors, device):
    """Transfer tensor, dict or list of tensors to device.

    Args:
        tensors (:class:`torch.Tensor`): May be a single, a list or a
            dictionary of tensors.
        device (:class: `torch.device`): the device where to place the tensors.

    Returns:
        Union [:class:`torch.Tensor`, list, tuple, dict]:
            Same as input but transferred to device.
            Goes through lists and dicts and transfers the torch.Tensor to
            device. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (list, tuple)):
        return [tensors_to_device(tens, device) for tens in tensors]
    elif isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = tensors_to_device(tensors[key], device)
        return tensors
    else:
        return tensors


def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode("utf-8")[:-1]
    return out


def get_free_gpu_indices():
    out = run_cmd("nvidia-smi -q -d Memory | grep -A4 GPU")
    out = (out.split("\n"))[1:]
    out = [l for l in out if "--" not in l]

    total_gpu_num = int(len(out) / 5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i * 5 : i * 5 + 1]][0])

    out = run_cmd("nvidia-smi --query-compute-apps=gpu_bus_id --format=csv")
    gpu_bus_ids_in_use = (out.split("\n"))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]
