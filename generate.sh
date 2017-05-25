#!/bin/bash
#fairseq optimize-fconv -input_model trainings/fconv/model_best.th7 -output_model trainings/fconv/model_best_opt.th7

curl https://s3.amazonaws.com/fairseq/data/wmt14.en-fr.newstest2014.tar.bz2 | tar xvjf -

fairseq generate -sourcelang en -targetlang fr -datadir data-bin/wmt14.en-fr -dataset newstest2014 \
  -path wmt14.en-fr.fconv-cuda/model.th7 -beam 5 -batchsize 128 | tee /tmp/gen.out
