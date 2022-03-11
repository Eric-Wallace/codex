#!/bin/bash

# model="/checkpoint/dpf/models/gpt-j/gpt-j-6B/"
# prefix="gpt-j"

model="/checkpoint/dpf/models/gpt-j/gpt-j-6B_ours62001/"
prefix="gpt-j-ft62001"

# iteration=29000
# model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/${iteration}.pt"

name=$1
if [ -z $name ]
then
  name="default"
else
  shift
fi
temperature=0.2

out_dir=out/${prefix}_ncg-1_temp-${temperature}/${name}

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
