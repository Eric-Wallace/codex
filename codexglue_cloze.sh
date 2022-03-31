#!/bin/bash

score_method=$1 # lr or inf

# model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt
# model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_36750_consolidated.pt"
model="/checkpoint/dpf/models/cm-1.3B/checkpoint_last_consolidated.pt"

# cloze mode must coordinate with path
cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-maxmin/"
# cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-all/"

outdir="out/codexgluecloze"
mkdir -p $outdir

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
  | tee ${outdir}/log.out
