#!/bin/bash

model="/checkpoint/dpf/models/gpt-j/gpt-j-6B/"

# iteration=29000
# model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/${iteration}.pt"

name=$1
if [ -z $name ]
then
  name="default"
else
  shift
fi

temperature=0.2

out_dir=mbpp/gpt-j_ncg-1_temp-${temperature}/${name}
#out_dir=/scratch/dpf/out

mkdir -p $out_dir

python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --num_candidates_evaluated 1 \
  --temperature $temperature \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
