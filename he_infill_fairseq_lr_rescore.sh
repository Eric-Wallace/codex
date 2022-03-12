#!/bin/bash

scoring=$1
num_candidates=$2
temperature=$3

name="cm-6B-armen_last_lr-gen_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"

outdir="expts/he_infill/${name}"
mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name /checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/best.pt \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring $scoring \
  --bidirectional_scoring \
  --batch_size 10 \
  --eval_type one_line \
  --truncation_heuristics num_lines suffix \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --result_base_path ${outdir}/results \
  | tee ${outdir}/log.out
