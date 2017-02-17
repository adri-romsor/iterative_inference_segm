#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS='device=gpu'
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python train_dae.py
