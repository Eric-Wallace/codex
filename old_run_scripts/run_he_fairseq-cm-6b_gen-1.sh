#!/bin/bash
iters=best
#model="/checkpoint/armenag/codex/models/checkpoint_1_${iters}_consolidated.pt"
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_${iters}_consolidated.pt"

name=$1
shift

temperature=0.2

#out_dir=out/cm-6B-fp16_it-${iters}_ncg-1_temp-${temperature}/${name}
out_dir=out/cm-6B_it-${iters}_ncg-1_temp-${temperature}/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --temperature $temperature \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
