#!/bin/bash
paths=trainings/fconvzh

fairseq optimize-fconv -input_model $paths/model_best.th7 -output_model $paths/model_best_opt.th7
DATA=data-bin/iwslt14.tokenized.zh-en
th generate-lines-zh.lua -sourcedict $DATA/dict.zh.th7 -targetdict $DATA/dict.en.th7 \
  -path $paths/model_best_opt.th7 -beam 3 -nbest 1 -input /root/fairseq/data/prep/tmp/train.tags.zh-en.zh
  #/home/pan/download/zh-en/valid-zh-en-t.zh