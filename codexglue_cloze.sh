#!/bin/bash

score_method=$1 # lr or inf or codex
# model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_36750_consolidated.pt"
# model="/checkpoint/dpf/models/cm-1.3B/checkpoint_last_consolidated.pt"

# cloze mode must coordinate with path
cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-maxmin/"
# cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-all/"

hash=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 8)
outdir=out-$1-$hash
mkdir -p $outdir

# OpenAI model names: code-davinci-002, code-davinci-001
# model="code-davinci-001"

python codexgluecloze.py \
  --git_status \
  --model_name $model \
  --tokenizer_name gpt2_pretokenization_newlines_only  \
  --candidate_scoring random \
  --batch_size 10 \
  --truncation_heuristics comment \
  --temperature 1 \
  --num_candidates 1 \
  --bidirectional_generation \
  --resume \
  --cloze_path $cloze_path \
  --cloze_mode maxmin \
  --score_method $score_method \
  --output_dir $outdir \
  --leftpad 20 \
  --rightpad 20 \
  | tee ${outdir}/log.out
