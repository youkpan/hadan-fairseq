#!/bin/bash
source /root/.bashrc
export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/nccl/build/lib/:/usr/local/cuda/lib64/
export CUDA_VISIBLE_DEVICES=0

th train.lua -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model fconv -nenclayer 4 -batchsize 8 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv -ngpus 1
