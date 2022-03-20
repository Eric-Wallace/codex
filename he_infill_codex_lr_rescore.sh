#!/bin/bash

scoring=$1
num_candidates=$2
temperature=$3

name="code-davinci-001_lr-gen_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"
#name="davinci-codex_lr-gen_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"

outdir="expts/he_infill/${name}"
mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name code-davinci-001 \
  --candidate_scoring $scoring \
  --bidirectional_scoring \
  --batch_size 10 \
  --eval_type one_line \
  --truncation_heuristics num_lines suffix \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --result_base_path ${outdir}/results \
  --max_tokens 60 \
  | tee ${outdir}/log.out
