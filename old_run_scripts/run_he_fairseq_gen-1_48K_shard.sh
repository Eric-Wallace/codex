#!/bin/bash
#model="/checkpoint/dpf/models/lm-1.3B-ourtok-lr1.6e-3/48000.pt"
model="/checkpoint/dpf/2021-12-24/code.tk-ours.1.3b.fsdp.me_fp16.transformer_lm_gpt.nlay24.emb2048.bm_none.tps2048.adam.fp16adam.b2_0.98.eps1e-08.cl0.0.lr0.0016.wu4700.dr0.1.atdr0.1.wd0.01.ms8.uf2.mu49591.s1.ngpu32/checkpoint_1_48000-shard0.pt"

name=$1
shift

out_dir=out/ncg-1_48K_shard/${name}

mkdir -p $out_dir

python -u he.py \
  --model_name $model \
  --num_candidates_generated 1 \
  --temperature 0.2 \
  --top_p 0.95 \
  --output_filename $out_dir/samples.jsonl \
  --response_filename $out_dir/responses.pkl \
  "$@" \
  | tee $out_dir/log.out
