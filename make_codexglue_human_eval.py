import os
from datasets import load_dataset
import csv
import json
import random

num_eval_samples = 200

prefix = "/shared/dfried/code_models/codexglue_code_to_text_multiline"
models = {
        "infill": "test_cm-6B_ncg-1_temp-0.0",
        "lr-rerank": "test_cm-6B_lr_rescore_scoring-sum_ncg-10_temp-0.2",
        "lr-single": "test_cm-6B_lr_rescore_scoring-mean_ncg-1_temp-0.0"
        }

# cat all shard files
for model, model_dir in models.items():
    os.system(f"cat {prefix}/{model_dir}/shard*/results*txt > data/docstring_human_eval/{model}.txt")

model_outputs = {
        model: [line.strip() for line in open(f"data/docstring_human_eval/{model}.txt", "r").readlines()] for model in models.keys()
        }
original_code = load_dataset("code_x_glue_ct_code_to_text", "python", split="test") 

# Randomly select indices
selected_idxs = list(sorted(random.sample(range(len(original_code)), num_eval_samples)))

# Remove old data
#os.system("rm data/docstring_human_eval/*.csv")
os.system("rm data/docstring_human_eval/*json")

# Write csv file for human eval
#writer = csv.writer(open("data/docstring_human_eval/human_eval.csv", "w"))
#writer.writerow(["id", "Original Code", "Model 1", "Model 2", "Model 3"])
writer = open("data/docstring_human_eval/human_eval.md", "w")

model_orders = []

for idx in selected_idxs:
    order = list(model_outputs.keys())
    random.shuffle(order)
    outputs = [model_outputs[model][idx].split("\t")[1].replace("\\n", "\n") for model in order]
    writer.write(f"""

# Example {idx}

Original Code:
```python
{original_code[idx]["code"]}
```
    
### Model 1:
```python
{outputs[0]}
```

### Model 2:
```python
{outputs[1]}
```

### Model 3:
```python
{outputs[2]}
```""")
#    writer.writerow([
#        idx,
#        original_code[idx]["code"].replace("\n", "\\n"),
##        original_code[idx]["docstring"].replace("\n", "\\n"),
#        model_outputs[order[0]][idx],
#        model_outputs[order[1]][idx],
#        model_outputs[order[2]][idx]
#        ])
    model_orders.append(order)

with open("data/docstring_human_eval/selected_idxs.json", "w") as f:
    json.dump(selected_idxs, f)
with open("data/docstring_human_eval/model_orders.json", "w") as f:
    json.dump(model_orders, f)

writer.close()
# Save:
# - output csv for eval
# - indices we selected
# - model shuffle order for each example
