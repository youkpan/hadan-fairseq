#!/bin/bash
#fairseq optimize-fconv -input_model trainings/fconv/model_best.th7 -output_model trainings/fconv/model_best_opt.th7
DATA=data-bin/iwslt14.tokenized.de-en
fairseq generate-lines -sourcedict $DATA/dict.de.th7 -targetdict $DATA/dict.en.th7 \
  -path trainings/fconv/model_best_opt.th7 -beam 10 -nbest 4