#!/bin/bash

conda activate base
conda remove -n av --all --yes
conda env create -f setup/requirements.yaml
conda activate av