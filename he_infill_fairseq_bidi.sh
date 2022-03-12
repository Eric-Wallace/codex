#!/bin/bash

num_candidates=$1
temperature=$2

beam=$3

if [ -z $beam ]
then
  beam=1
fi

name="cm-6B-armen_last_bidi-gen_ncg-${num_candidates}_temp-${temperature}_no-samp_beam-${beam}"

outdir="expts/he_infill/${name}"
mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name /checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/best.pt \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring random \
  --batch_size 10 \
  --eval_type one_line \
  --truncation_heuristics num_lines suffix \
  --temperature ${temperature} \
  --beam ${beam} \
  --num_candidates ${num_candidates} \
  --bidirectional_generation \
  --result_base_path ${outdir}/results \
  | tee ${outdir}/log.out
