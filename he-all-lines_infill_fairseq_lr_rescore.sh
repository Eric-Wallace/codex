#!/bin/bash

scoring=$1
num_candidates=$2
temperature=$3
suffix=$4
shard=$5
shift
shift
shift
shift
shift

name="cm-6B-armen_last_lr-gen_pyprefix_scoring-${scoring}_ncg-${num_candidates}_temp-${temperature}_${suffix}"
outdir="expts/he_infill_all/${name}"
outdir=${outdir}/shard_${shard}

mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name /checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/best.pt \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring $scoring \
  --bidirectional_scoring \
  --batch_size 10 \
  --eval_type all_lines \
  --max_tokens 450 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --result_base_path ${outdir}/results \
  --prompt_prefix "<| file ext=.py |>" \
  --shard_number $shard \
  $@ \
  | tee ${outdir}/log.out
