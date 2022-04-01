#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"

split=$1
temperature=$2
num_candidates=1

outdir="expts/crt/${split}_cm-6B_indent_def_ncg-${num_candidates}_temp-${temperature}_prompt-py"

mkdir -p $outdir

python codexglue_return_types.py \
  --split $split \
  --git_status \
  --model_name ${model} \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --bidirectional_generation \
  --max_tokens 20 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --truncation_heuristics  \
  --result_base_path $outdir/results \
  --prompt_prefix "<| file ext=.py |>" \
  | tee ${outdir}/log.out
