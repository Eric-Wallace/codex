#!/bin/bash
prefix="scratch"
model="/checkpoint/dpf/models/lm-1.3B-gpt2tok/20000.pt"

# prefix="xlmg"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-xlmg/20000.pt"

name=$1
shift

#out_dir=mbpp/ncg-1_48K/${name}
out_dir=mbpp/lm-1B-gpt_${prefix}_ncg-1_temp-0.2/${name}

mkdir -p $out_dir

python -u mbpp.py \
  --model_name $model \
  --tokenizer_name gpt2 \
  --num_candidates_generated 1 \
  --num_candidates_evaluated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
