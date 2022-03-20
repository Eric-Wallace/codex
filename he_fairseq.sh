#!/bin/bash

model_name=$1
shift

if [ "$model_name" == "cm-6B-armen" ]
then
  batch_size=10
else
  batch_size=20
fi

model="/checkpoint/dpf/models/${model_name}/checkpoint_last_consolidated.pt"

ncg=$1
shift

temperature=$1
shift

name=$1
shift

out_dir=expts/he/${model_name}_last_pg_ncg-${ncg}_temp-${temperature}/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated ${ncg} \
  --num_candidates_evaluated 1 \
  --batch_size $batch_size \
  --temperature $temperature \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
