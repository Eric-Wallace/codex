#!/bin/bash
model="/checkpoint/dpf/models/lm-1.3B-ourtok-lr1.6e-3/38000.pt"

name=$1
shift

out_dir="out/38K/${name}"

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 15 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
