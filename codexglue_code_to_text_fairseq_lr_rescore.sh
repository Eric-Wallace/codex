#!/bin/bash

scoring=$1
num_candidates=$2
temperature=$3

model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt

outdir="out/codexglue_code_to_text/lr_rescore__scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"
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
  | tee ${outdir}/log.out
