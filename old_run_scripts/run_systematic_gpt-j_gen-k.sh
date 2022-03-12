#!/bin/bash

model="/home/jessy/projects/codex/gpt-j-6B_ours62001"
prefix="gpt-j"

# model="/checkpoint/dpf/models/gpt-j/gpt-j-6B_ours62001/"
# prefix="gpt-j-ft62001"

# iteration=29000
# model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/${iteration}.pt"

ncg=15
#shift

temperature=0.6
#shift

name=he_systematic_gen1
#if [ -z $name ]
#then
#  name="default"
#else
#  shift
#fi

out_dir=out/${prefix}_ncg-${ncg}_temp-${temperature}/${name}

mkdir -p $out_dir

python -u infill_evaluation.py \
  --model_path $model \
  --num_candidates 1 \
  --result_base_path $out_dir \
  --batch_size 3 \
  --temperature $temperature \
  --top_p 0.95 \
  "$@" \
  | tee $out_dir/log.out
