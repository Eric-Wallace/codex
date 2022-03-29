#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"
#model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt

num_candidates=$1
temperature=$2
split=$3
if [ -z $split ]
then
  split="test"
fi

outdir="expts/codexglue_code_to_text/${split}_cm-6B_lr_ncg-${num_candidates}_temp-${temperature}"

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
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring random \
  --batch_size 10 \
  --truncation_heuristics comment \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --result_base_path ${outdir}/results \
  --split $split \
  --shard_number $shard \
  | tee ${outdir}/log.out
