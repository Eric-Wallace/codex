#!/bin/bash
model_name=$1
shift

model="/checkpoint/dpf/models/${model_name}/checkpoint_last_consolidated.pt"

ncg=$1
shift

temperature=$1
shift

name=$1
shift

split=$1
if [ -z $split ]
then
  split="validation"
else
  shift
fi

out_dir=expts/mbpp/${split}_${model_name}_last_pg_ncg-${ncg}_temp-${temperature}/${name}

mkdir -p $out_dir

python -u mbpp.py \
  --split $split \
  --model_name $model \
  --num_candidates_generated ${ncg} \
  --num_candidates_evaluated 1 \
  --temperature $temperature \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
