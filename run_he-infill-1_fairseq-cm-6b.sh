#!/bin/bash
iters=best
#model="/checkpoint/armenag/codex/models/checkpoint_1_${iters}_consolidated.pt"
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_${iters}_consolidated.pt"

temperature=$1
shift

name=$1
if [[ -z $name ]]
then
  name="default"
else
  shift
fi

out_dir=out_infill/cm-6B_it-${iters}_temp-${temperature}/${name}

mkdir -p $out_dir

python -u infill_evaluation.py \
  --model_path $model \
  --eval_type one_line \
  --temperature $temperature \
  --top_p 0.95 \
  --result_base_path $out_dir/one_line \
  "$@" \
  | tee $out_dir/log.out
