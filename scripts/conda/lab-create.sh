#!/usr/bin/env bash

CONDA_PREFIX='envs/lab'

conda create -y -p ${CONDA_PREFIX} python=3.11

conda install -y -p ${CONDA_PREFIX} \
    tqdm pytest requests ftfy ipykernel ipywidgets


