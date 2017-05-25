#!/bin/bash
paths=trainings/fconvzh-back

fairseq optimize-fconv -input_model $paths/model_best.th7 -output_model $paths/model_best_opt.th7
DATA=data-bin/iwslt14.tokenized.zh-zh
th generate-back-zh.lua -sourcedict $DATA/dict.zh.th7 -targetdict $DATA/dict.en.th7 \
  -path $paths/model_best_opt.th7   -beam 5 -nbest 4 -input /home/pan/download/zh-en/valid-zh-en-t.zh
  #\
  #-input /root/fairseq/data/prep/tmp/train.tags.zh-en.zh
  #-input /root/fairseq/data/prep/tmp/train.tags.zh-en.zh
  #/home/pan/download/zh-en/valid-zh-en-t.zh