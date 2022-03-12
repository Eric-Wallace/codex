#!/bin/bash

# iter=last
# model="/checkpoint/dpf/models/lm-1.3B-tk-ours-py-lr4e-4/${iter}.pt"
# out_dir="out/lm-1.3B-tk-ours-py-lr4e-4_it-${iter}/${name}"

iter=last
model="/checkpoint/dpf/models/lm-1.3B-tk-ours-py-lr8e-4/${iter}.pt"
out_dir="out/lm-1.3B-tk-ours-py-lr8e-4_it-${iter}/${name}"

name=$1
shift


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

./evaluate_he.sh ${out_dir}/samples.jsonl | tee ${model%.*}-humaneval-gen.out
