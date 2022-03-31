from datasets import load_dataset
import sys
from mosestokenizer import MosesDetokenizer

def make_codexglue_reference_file(output_file: str, split="test", eval_type="cleaned"):
    assert eval_type in ["full", "cleaned"]
    ds = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)

    if eval_type == "cleaned":
        f_detok = open(output_file + ".detok", "w")
        detokenize = MosesDetokenizer("en")

    with open(output_file, "w") as f:
        for i in range(len(ds)):
            # Eval with true docstrs
            if eval_type == "full":
                gold = ds[i]["docstring"].encode("unicode_escape").decode("utf-8")
                f.write(f"{i}\t{gold}\n")

            # Eval with clean tokenized docstrings (used in paper)
            elif eval_type == "cleaned":
                #  these postprocessing steps are the ones the authors use,
                gold=' '.join(ds[i]['docstring_tokens']).replace('\n','')
                gold=' '.join(gold.strip().split())
                # apply our own detokenizer
                gold_detok = detokenize(gold.split())
                f.write(f"{i}\t{gold}\n")
                f_detok.write(f"{i}\t{gold_detok}\n")

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
