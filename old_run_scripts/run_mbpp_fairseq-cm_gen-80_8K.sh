#!/bin/bash
model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4--4e-4/34000.pt"

name=$1
shift

out_dir=mbpp/cm_ncg-80_34K_0.2/${name}

mkdir -p $out_dir

for seed in 1
do
python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 80 \
  --num_candidates_evaluated 80 \
  --temperature 0.5 \
  --top_p 0.95 \
  --output_path $out_dir/outputs_s-${seed} \
  --verbose \
  "$@" \
  | tee $out_dir/log_s-${seed}.out
done
