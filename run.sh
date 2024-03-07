#!/bin/bash

train_and_evaluate() {
    local dataset=$1
    local config="config/$2.yaml"
    local exp="../experiments/audio-visual/$3/conf.yaml"
    local test_dir="data-preprocess/$1/tt"

    echo "Stage 1: Training: python train.py --conf-dir $config"
    python train.py --conf-dir $config

    echo "Stage 2: Evaluation: python test.py --conf-dir $exp --test-dir $test_dir"
    python test.py --conf-dir $exp --test-dir $test_dir
}

# RTFSNet
# 4 layers
train_and_evaluate "LRS2" "lrs2_RTFSNet_4_layer" "RTFS-Net/LRS2/4_layers"
train_and_evaluate "LRS3" "lrs3_RTFSNet_4_layer" "RTFS-Net/LRS3/4_layers"
train_and_evaluate "VOX2" "voxceleb2_RTFSNet_4_layer" "RTFS-Net/VOX2/4_layers"

# 6 layers
train_and_evaluate "LRS2" "lrs2_RTFSNet_6_layer" "RTFS-Net/LRS2/6_layers"
train_and_evaluate "LRS3" "lrs3_RTFSNet_6_layer" "RTFS-Net/LRS3/6_layers"
train_and_evaluate "VOX2" "voxceleb2_RTFSNet_6_layer" "RTFS-Net/VOX2/6_layers"

# 12 layers
train_and_evaluate "LRS2" "lrs2_RTFSNet_12_layer" "RTFS-Net/LRS2/12_layers"
train_and_evaluate "LRS3" "lrs3_RTFSNet_12_layer" "RTFS-Net/LRS3/12_layers"
train_and_evaluate "VOX2" "voxceleb2_RTFSNet_12_layer" "RTFS-Net/VOX2/12_layers"