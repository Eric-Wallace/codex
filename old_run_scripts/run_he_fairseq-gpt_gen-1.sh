#!/bin/bash

# prefix="scratch"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok/20000.pt"

# prefix="xlmg"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-xlmg/40000.pt"

# prefix="fixdata-scratch-20K"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata/20000.pt"

# prefix="fixdata-scratch-40K"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata/40000.pt"

# prefix="fixdata-xlmg-20K"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata-xlmg/20000.pt"

# prefix="fixdata-xlmg-40K"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata-xlmg/40000.pt"

# prefix="fixdata-xlmg-last"
# model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata-xlmg/last.pt"

prefix="fixdata-scratch-last"
model="/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata/last.pt"


name=$1
shift

out_dir=out/lm-1B-gpt_${prefix}_ncg-1_temp-0.2/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --tokenizer_name gpt2 \
  --num_candidates_generated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
