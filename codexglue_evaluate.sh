#!/bin/bash

root_dir=$1

for f in $root_dir/shard_*/results_shard*-of-10.txt
do
    python codexglue_eval.py $f
done

cat $root_dir/shard_{0,1,2,3,4,5,6,7,8,9}/results_shard-*-of-10.txt.postprocessed > $root_dir/results.txt.postprocessed

python codexglue_bleu_evaluator.py data/code_summ_clean_ref.detok.txt < $root_dir/results.txt.postprocessed
