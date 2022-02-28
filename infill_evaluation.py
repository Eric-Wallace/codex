from human_eval.data import read_problems
from human_eval.execution import check_correctness
from utils import build_systematic_infill_prompt
import json
import pickle
import numpy as np
import tqdm
import os
import sys
import pprint

from causal_masking_infill import InfillingModel

from he import HUMAN_EVAL_STOP_WORDS

def run_systematic_infill(args, eval_type="one_line", result_base_path=None):
    """Masks out a subset of lines in the HumanEval reference solution and infills with the CM model. Saves the output to a file for later evaluation.

    Args:
        eval_type (str):
            "one_line": mask out all possible single lines
            "all_lines": mask out all possible multi-line chunks
    """
    infilling_model = InfillingModel(args.model_path)

    assert eval_type in ("one_line", "all_lines")
    problems = list(sorted(read_problems().items()))
    results = []
    if result_base_path is not None:
        result_json_fname = f"{result_base_path}.json"
        result_pkl_fname = f"{result_base_path}.pkl"
    else:
        result_json_fname = f"he_{eval_type}_systematic.json"
        result_pkl_fname = f"he_{eval_type}_systematic.pkl"
    result_json = open(result_json_fname, "w")

    for i, (task_id, problem) in enumerate(tqdm.tqdm(problems, ncols=120)): 
        soln = problem["canonical_solution"].rstrip() # note we strip extra newlines
        num_lines = len(soln.split("\n"))
        
        infill_results = []
        num_lines_to_mask = []        
        if eval_type == "one_line":
            for num_before in range(0, num_lines):
                num_lines_to_mask.append((num_before, num_lines - num_before - 1))
        else:
            for num_before in range(0, num_lines):
                for num_after in range(0, num_lines - num_before):
                    num_lines_to_mask.append((num_before, num_after))

        for num_before, num_after in num_lines_to_mask:
            prompt_parts, missing_lines = build_systematic_infill_prompt(
                    problem["prompt"],
                    soln,
                    num_before,
                    num_after)
            if args.temperature == 0.0:
                out = infilling_model.infill(prompt_parts, verbose=False, sampling=False)
            else:
                out = infilling_model.infill(prompt_parts, verbose=False, sampling=True, sampling_topp=args.top_p, sampling_temperature=args.temperature)
            infill_results.append({
                "num_before": num_before,
                "num_after": num_after,
                "missing_lines": missing_lines, 
                "prompt_parts": prompt_parts,
                "infill": out["infills"][0],
                "complete": out["complete"][0],
                })
            # print("="*20)
            # print("Prompt: ", prompt_parts)
            # print(out["infills"][0])

        result = {
                "num_lines": num_lines,
                "task_id": task_id,
                "canonical_solution": problem["canonical_solution"],
                "infill_results": infill_results,
                }
        results.append(result)
        result_json.write(json.dumps(result) + "\n")
        if i % 10 == 0:
            result_json.flush()

    with open(result_pkl_fname, "wb") as f:
        pickle.dump(results, f)

    result_json.close()

def read_file(filename):
    if filename.endswith(".json"):
        with open(filename) as f:
            return [json.loads(line) for line in f]
    elif filename.endswith(".pkl"):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        raise NotImplementedError()

def evaluate_one_line_systematic(filename):
    """Reads infilled one-line completions from a file, postprocesses them (truncates them to one line), and evaluates
    functional correctness (average pass rate / exact match).
    """

    outputs = read_file(filename)
    problems = read_problems()

    functional_results = []
    samples_to_eval = []
    with tqdm.tqdm(outputs, ncols=120) as p:
        for out in p:
            if functional_results:
                avg_pass = np.mean([x["passed"] for x in functional_results])
                avg_exact = np.mean([x["exact_match"] for x in functional_results])
                p.set_postfix({'pass': avg_pass, 'exact': avg_exact})
            for infill_res in out["infill_results"]:
                # truncate infill to be one line
                infill = infill_res["infill"]
                if infill.startswith("\n"):
                    if "\n" not in infill[1:]:
                        infilled_line = infill
                    else:
                        infilled_line = infill[:infill[1:].index("\n") + 1]
                else:
                    if "\n" in infill:
                        infilled_line = infill[:infill.index("\n") + 1]
                    else:
                        infilled_line = infill

                # check mismatched indent?
                is_exact_match = infilled_line.rstrip() == infill_res["missing_lines"].rstrip()

                complete = "".join([infill_res["prompt_parts"][0],
                    infilled_line,
                    infill_res["prompt_parts"][1]])

                res = check_correctness(
                    problem=problems[out["task_id"]],
                    completion=complete,
                    timeout=3.0)
                functional_results.append({
                    "task_id": out["task_id"],
                    "num_before": infill_res["num_before"],
                    "num_after": infill_res["num_after"],
                    "passed": res["passed"],
                    "exact_match": is_exact_match, 
                    })
                #print(f"{out['task_id']} | pass {res['passed']} | exact {is_exact_match}")

    avg_pass = np.mean([x["passed"] for x in functional_results])
    avg_exact = np.mean([x["exact_match"] for x in functional_results])
    
    print("average pass:", avg_pass)
    print("average exact:", avg_exact)

    ext_stripped = os.path.splitext(filename)[0]
    with open(f"{ext_stripped}__functional_eval.json", "w") as f:
        json.dump(functional_results, f)

if __name__ == "__main__":
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_base_path")
    parser.add_argument("--eval_type", choices=["one_line", "all_lines"], default="one_line")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    if not args.evaluate_only:
        run_systematic_infill(args, eval_type=args.eval_type, result_base_path=args.result_base_path)
    evaluate_one_line_systematic(f"{result_base_path}.pkl")
