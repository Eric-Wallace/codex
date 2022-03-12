#!/bin/bash

iter=40000
model="/checkpoint/dpf/models/lm-1.3B-ourtok-lr1.6e-3/${iter}.pt"

name=$1
shift

out_dir=out_ncg-1/${name}
out_dir="out/ncg-1_it-${iter}/${name}"

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
