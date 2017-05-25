#!/bin/bash
source /root/.bashrc
export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/nccl/build/lib/:/usr/local/cuda/lib64/
export CUDA_VISIBLE_DEVICES=0

TEXT=data/iwslt14.tokenized.zh-en

fairseq preprocess -sourcelang zh -targetlang en \
  -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
  -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.zh-en