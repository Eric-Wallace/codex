#!/bin/bash

# model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"
short_model=$1
shift
model="bigcode/${short_model}"

num_candidates=$1
temperature=$2
split=$3
if [ -z $split ]
then
  split="test"
fi

outdir="expts/codexglue_code_to_text/${split}_${short_model}_no-file_ncg-${num_candidates}_temp-${temperature}"

shard=$4
if [ -z $shard ]
then
  shard=-1
else
  outdir=${outdir}/shard_${shard}
fi

mkdir -p $outdir

python codexglue.py \
  --git_status \
  --model_name $model \
  --candidate_scoring random \
  --batch_size 10 \
  --truncation_heuristics comment \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --bidirectional_generation \
  --result_base_path ${outdir}/results \
  --split $split \
  --shard_number $shard \
  --max_input_length 2048 \
  | tee ${outdir}/log.out
