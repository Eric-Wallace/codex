#!/bin/bash

iteration=49000
model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4--4e-4/${iteration}.pt"

# iteration=29000
# model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/${iteration}.pt"

name=$1
shift

out_dir=mbpp/cm_it-${iteration}_ncg-1_0.2/${name}

mkdir -p $out_dir

python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --num_candidates_evaluated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
