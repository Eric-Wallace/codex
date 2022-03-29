#!/bin/bash

num_candidates=$1
temperature=$2
suffix=$3
shard=$4
shift
shift
shift
shift

name="cm-6B-armen_last_bidi-gen_pyprefix_ncg-${num_candidates}_temp-${temperature}_${suffix}"
outdir="expts/he_infill_all/${name}"
#outdir=${outdir}/shard_${shard}

mkdir -p $outdir

python infill_evaluation.py \
  --git_status \
  --model_name /checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/best.pt \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring random \
  --batch_size 10 \
  --eval_type all_lines \
  --max_tokens 450 \
  --temperature ${temperature} \
  --num_candidates ${num_candidates} \
  --bidirectional_generation \
  --result_base_path ${outdir}/results \
  --prompt_prefix "<| file ext=.py |>" \
  --shard_number $shard \
  $@ \
  | tee ${outdir}/log_shard-${shard}.out
