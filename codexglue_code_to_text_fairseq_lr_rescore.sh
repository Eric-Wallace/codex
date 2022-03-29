#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"
#model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt

scoring=$1
num_candidates=$2
temperature=$3
split=$4
if [ -z $split ]
then
  split="test"
fi

outdir="expts/codexglue_code_to_text/${split}_cm-6B_lr_rescore_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"

shard=$5
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
  --candidate_scoring $scoring \
  --bidirectional_scoring \
  --batch_size 10 \
  --truncation_heuristics comment \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --result_base_path ${outdir}/results \
  --split $split \
  --shard_number $shard \
  | tee ${outdir}/log.out
