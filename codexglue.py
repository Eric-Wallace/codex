from typing import List

from datasets import load_dataset

from utils import split_function_into_docstring_infill_prompt, TRIPLE_QUOTE
from causal_masking_infill import infill

class CodeDataset:

    def __init__(self):
        # load data
        self.data = ...

    def get_prompt_parts(self, i: int) -> List[str]:
        """Gets the text prompt for example i.

        Returns:
            list of prompt parts (where the model must fill in text between each part, for cloze-type tasks)
            if just a "vanilla" prompt, a list of length 1
        """
        raise NotImplementedError()

    def evaluate(self, i: int, model_completion: str):
        """Given the model completion for example i, returns a bool indicating whether
        the problem was solved."""
        raise NotImplementedError()

class CodeXGlueClozeAllDataset(CodeDataset):
    
    def __init__(self):
        self.data = load_dataset("code_x_glue_cc_cloze_testing_all", "python", split="train")

    def get_prompt_parts(self, i):
        pass
#        example = self.data[i]
#        code_toks = example["pl_tokens"]
#        docstring = " ".join(example["nl_tokens"])
#
#        function_def = " ".join(code_toks[:code_toks.index(":") + 1])
#        code_toks = code_toks[code_toks.index(":") + 1:]
#        body_pre_mask = " ".join(
#                code_toks[code_toks.index(":") + 1:code_toks.index("<mask>")])
#
#        prompt_prefix = f"{function_def}\n    {TRIPLE_QUOTE}{docstring}{TRIPLE_QUOTE}\n{body_pre_mask}"
#        prompt_suffix = " ".join(code_toks[code_toks.index("<mask>") + 1:])
#        return [prompt_prefix, prompt_suffix]

class CodeXGlueCodeSummDataset(CodeDataset):

    def __init__(self):
        self.data = load_dataset("code_x_glue_ct_code_to_text", "python", split="test")

    def get_prompt_parts(self, i):
        return split_function_into_docstring_infill_prompt(self.data[i]["original_string"], has_docstring=True)

    def evaluate(self, i, model_completion):
        pass

if __name__ == "__main__":
    ds = CodeXGlueCodeSummDataset()

    outputs = []

    for i in range(len(ds.data)):
        try:
            out = infill(ds.get_prompt_parts(i), verbose=False, sampling=False)
            pred_infill = out["infills"][0]
            if TRIPLE_QUOTE not in pred_infill:
                print(f"ERROR on {i}: ", pred_infill)
                docstr = pred_infill
            else:
                docstr = pred_infill[:pred_infill.index(TRIPLE_QUOTE)].strip() 
        except Exception as e:
            print(e)
            docstr = ""
            import pdb; pdb.set_trace()
        outputs.append(docstr)
        print(i)
        print(docstr)

    with open("code_summ_preds_greedy.pkl", "wb") as f:
        pickle.dump(outputs, f)

    with open("code_summ_preds_greedy.txt", "w") as f:
        for i, output in enumerate(outputs):
            f.write(f"{i}\t{output}")
