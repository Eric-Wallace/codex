#!/bin/bash

num_candidates=$1
temperature=$2
suffix=$3
shift
shift
shift

name="code-davinci-002_bidi_ncg-${num_candidates}_temp-${temperature}_${suffix}"
#name="davinci-codex_lr-gen_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}"

outdir="expts/he_infill/${name}"
mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name code-davinci-002 \
  --candidate_scoring random \
  --batch_size 10 \
  --eval_type one_line \
  --max_tokens 60 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --bidirectional_generation \
  --result_base_path ${outdir}/results \
  $@ \
  | tee ${outdir}/log.out
