#!/bin/bash
iters=6000
model="/checkpoint/dpf/models/cm-6B/checkpoint_1_${iters}_consolidated.pt"

name=$1
shift

out_dir=mbpp/cm-6B_it-${iters}_ncg-15/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 15 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
