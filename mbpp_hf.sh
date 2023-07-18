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

split=$1
if [ -z $split ]
then
  split="validation"
else
  shift
fi

out_dir=expts/mbpp/${split}_${short_model_name}_last_pg_ncg-${ncg}_temp-${temperature}/${name}

mkdir -p $out_dir

export TOKENIZERS_PARALLELISM=false

python -u mbpp.py \
  --git_status \
  --model_name $model_name \
  --split $split \
  --num_candidates_generated ${ncg} \
  --num_candidates_evaluated 1 \
  --batch_size $batch_size \
  --temperature $temperature \
  --top_p 0.95 \
  --output_path $out_dir/outputs \
  --verbose \
  "$@" \
  | tee $out_dir/log.out
