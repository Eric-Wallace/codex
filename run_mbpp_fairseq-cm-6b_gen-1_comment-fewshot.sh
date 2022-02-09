#!/bin/bash
iters=36750
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_${iters}_consolidated.pt"
#model="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4--4e-4/34000.pt"

name=$1
shift

k=$1
if [ -z $k ]
then
  k=3
else
  shift
fi

temperature=0.2

out_dir=mbpp/cm-6B_it-${iters}_comment-k-${k}_ncg-1_temp-${temperature}/${name}
#out_dir=/scratch/dpf/out

mkdir -p $out_dir

python -u mbpp.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --num_candidates_evaluated 1 \
  --temperature $temperature \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  --k_shot ${k} \
  --prompt_template comment \
  "$@" \
  | tee $out_dir/log.out
