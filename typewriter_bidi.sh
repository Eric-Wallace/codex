#!/bin/bash

model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_last_consolidated.pt"

num_candidates=1
temperature=0.0

suffix="_formatted_iandf"
#suffix="_formatted_full"

outdir="expts/typewriter/cm-6B${suffix}_indent_def_ncg-${num_candidates}_temp-${temperature}_prompt-py"
#outdir="expts/typewriter/cm-6B${suffix}_ncg-${num_candidates}_temp-${temperature}"

shard=$1
if [ -z $shard ]
then
  shard=-1
else
  outdir=${outdir}/shard_${shard}
fi

mkdir -p $outdir

python typewriter.py \
  data/typewriter_examples${suffix}.json \
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
  --prompt_prefix "<| file ext=.py |>" \
  | tee ${outdir}/log.out
