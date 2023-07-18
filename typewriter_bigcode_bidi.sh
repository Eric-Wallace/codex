#!/bin/bash

short_model=$1
shift

#short_model="santacoder"
#short_model="large-model"
model="bigcode/${short_model}"

num_candidates=1
temperature=0.0

suffix="_formatted_iandf"
#suffix="_formatted_full"

outdir="expts/typewriter/${short_model}_indent_def_ncg-${num_candidates}_temp-${temperature}"

shard=$1
if [ -z $shard ]
then
  shard=-1
else
  outdir=${outdir}/shard_${shard}
fi

mkdir -p $outdir

python typewriter.py \
  data/typewriter_examples${suffix}.json \
  --git_status \
  --model_name ${model} \
  --bidirectional_generation \
  --max_tokens 20 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --truncation_heuristics  \
  --result_base_path $outdir/results \
  --shard_number $shard \
  | tee ${outdir}/log.out
