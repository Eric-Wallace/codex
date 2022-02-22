#!/bin/bash
iters=36750
#model="/checkpoint/armenag/codex/models/checkpoint_1_${iters}_consolidated.pt"
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_${iters}_consolidated.pt"

ncg=$1
shift

temperature=$1
shift

name=$1
shift

out_dir=out/cm-6B_it-${iters}_ncg-${ncg}-1_temp-${temperature}/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated ${ncg} \
  --num_candidates_evaluated 1 \
  --batch_size 3 \
  --temperature $temperature \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out