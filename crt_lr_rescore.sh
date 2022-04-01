#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"

split=$1
scoring=$2
num_candidates=$3
temperature=$4
shift
shift
shift
shift

outdir="expts/crt/${split}_cm-6B_indent_def_lr_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}_prompt-py"

mkdir -p $outdir

python codexglue_return_types.py \
  --split $split \
  --git_status \
  --model_name ${model} \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring $scoring \
  --bidirectional_scoring \
  --max_tokens 20 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --truncation_heuristics  \
  --result_base_path $outdir/results \
  --prompt_prefix "<| file ext=.py |>" \
  | tee ${outdir}/log.out
