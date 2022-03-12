#!/bin/bash

iteration=49000
model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4--4e-4/${iteration}.pt"

# iteration=29000
# model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/${iteration}.pt"

name=$1
shift

out_dir=out/cm_it-${iteration}_ncg-1_temp-0.2/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
