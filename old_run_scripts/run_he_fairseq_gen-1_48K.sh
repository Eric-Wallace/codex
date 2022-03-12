#!/bin/bash
model="/checkpoint/dpf/models/lm-1.3B-ourtok-lr1.6e-3/48000.pt"

name=$1
shift

out_dir=out_ncg-1_48K/${name}

mkdir $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
