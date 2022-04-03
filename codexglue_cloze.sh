#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
## job name
#SBATCH --job-name=cloze
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/cloze-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u//cloze-%j.err


## configs
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 2
#SBATCH -C volta32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 20
#SBATCH --mem 120G 
#SBATCH --partition devlab 
#SBATCH --time 120 

# salloc --gpus-per-node 2 -C volta32gb --nodes 1 --ntasks-per-node 1 --cpus-per-task 20 --time 2500 --mem 120G --partition devlab

score_method=$1 # lr or inf or codex
# model=/home/jessy/projects/codex/evaluations/cm-6B-ourtok/38250.pt
model="/checkpoint/dpf/models/cm-6B-armen/checkpoint_1_36750_consolidated.pt"
# model="/checkpoint/dpf/models/cm-1.3B/checkpoint_last_consolidated.pt"

# cloze mode must coordinate with path
cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-maxmin/"
# cloze_path="/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-all/"
# module load anaconda3/5.0.1
source /private/home/sida/code-models-shared/fairseq-py/dpf_scripts/activate_htlm.sh
# hash=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 8)
echo $SLURM_JOB_ID
outdir=out-$1-$SLURM_JOB_ID
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
  --leftpad 10 \
  --rightpad 10 \
  | tee ${outdir}/log.out
