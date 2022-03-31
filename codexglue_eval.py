from datasets import load_dataset
import sys

def make_codexglue_reference_file(output_file: str, split="test"):
    ds = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)
    with open(output_file, "w") as f:
        print("new")
        for i in range(len(ds)):
            # Eval with true docstrs
            gold = ds[i]["docstring"].encode("unicode_escape").decode("utf-8")
            # Eval with clean tokenized docstrings (used in paper)
            #gold=' '.join(ds[i]['docstring_tokens']).replace('\n','')
            #gold=' '.join(gold.strip().split())
            f.write(f"{i}\t{gold}\n")

def postprocess_model_outputs(path_to_model_outputs_txt: str):
    with open(path_to_model_outputs_txt) as f, \
            open(path_to_model_outputs_txt + ".postprocessed", "w") as f_out:
        for line in f:
            idx, docstr = line.split("\t")
            docstr_one_line = docstr.split("\\n")[0].strip()
            f_out.write(f"{idx}\t{docstr_one_line}\n")

if __name__ == "__main__":
    model_outputs = sys.argv[1]
    postprocess_model_outputs(model_outputs)
