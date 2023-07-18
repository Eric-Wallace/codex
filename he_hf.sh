#!/bin/bash

model_name=$1
shift

batch_size=10

short_model_name=`basename $model_name`

ncg=$1
shift

temperature=$1
shift

name=$1
shift

out_dir=expts/he/${short_model_name}_last_pg_ncg-${ncg}_temp-${temperature}/${name}

mkdir -p $out_dir

export TOKENIZERS_PARALLELISM=false

python -u he.py \
  --git_status \
  --model_name $model_name \
  --num_candidates_generated ${ncg} \
  --num_candidates_evaluated 1 \
  --batch_size $batch_size \
  --temperature $temperature \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
