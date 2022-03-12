#!/bin/bash
model="/checkpoint/dpf/models/lm-1.3B-ourtok-lr1.6e-3/48000.pt"

name=$1
shift

out_dir=mbpp/ncg-1_48K/${name}

mkdir -p $out_dir

for seed in `seq 1 3`
do
python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --num_candidates_evaluated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_path $out_dir/outputs_s-${seed} \
  --verbose \
  "$@" \
  | tee $out_dir/log_s-${seed}.out
done
