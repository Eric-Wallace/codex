#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"

num_candidates=1
temperature=0.0

outdir="expts/codexglue_code_to_text/cm-6B_ncg-${num_candidates}_temp-${temperature}"

shard=$1
if [ -z $shard ]
then
  shard=-1
else
  outdir=${outdir}/shard_${shard}
fi

python typewriter.py \
  data/typewriter_examples.json \
  --git_status \
  --model_name ${model} \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --bidirectional_generation \
  --max_tokens 20 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --truncation_heuristics  \
  --result_base_path $outdir/results \
  --shard_number $shard \
  | tee ${outdir}/log.out
