#!/bin/bash

iters=best
model="/checkpoint/dpf/models/cm-6B/checkpoint_1_${iters}_consolidated.pt"

#model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4--4e-4/34000.pt"

name=$1
shift

out_dir=mbpp/cm-6B_it-${iters}_ncg-80_temp-0.5/${name}
#out_dir=/scratch/dpf/out

mkdir -p $out_dir

python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 80 \
  --num_candidates_evaluated 80 \
  --temperature 0.5 \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
