from human_eval.data import read_problems
from human_eval.execution import check_correctness
from utils import build_systematic_infill_prompt, truncate_num_lines, read_file, truncate_overlap, stripped_line_split
import json
import pickle
import numpy as np
import tqdm
import os
import sys
import pprint

from typing import List

from models import TruncationParameters, make_model, Model

from he import HUMAN_EVAL_STOP_WORDS, generate_he_infill_problems

def run_systematic_infill(args, model: Model, eval_type="one_line", result_base_path=None):
    """Masks out a subset of lines in the HumanEval reference solution and infills with the CM model. Saves the output to a file for later evaluation.

    Args:
        eval_type (str):
            "one_line": mask out all possible single lines
            "all_lines": mask out all possible multi-line chunks
    """

    assert eval_type in ("one_line", "all_lines")

    all_results = []
    if result_base_path is not None:
        result_json_fname = f"{result_base_path}.json"
        result_pkl_fname = f"{result_base_path}.pkl"
    else:
        result_json_fname = f"he_{eval_type}_systematic.json"
        result_pkl_fname = f"he_{eval_type}_systematic.pkl"
    result_json = open(result_json_fname, "w")

    problem_iterator = generate_he_infill_problems(args, eval_type)

    responses = {}

    for i, (task_id, task_id_problems) in enumerate(tqdm.tqdm(problem_iterator, ncols=120)): 
        
        infill_results = []
        for problem in task_id_problems:
            key = (task_id, problem["num_before"], problem["num_after"])

            prompt_parts = problem["prompt_parts"]
            if len(prompt_parts) != 2:
                raise NotImplementedError("multiple region infilling is not implemented")
            else:
                prefix, suffix = prompt_parts

            truncation_parameters = [
                TruncationParameters.from_heuristics(args.truncation_heuristics, problem["missing_lines"], suffix)
            ]
            kwargs = dict(
                verbose=False, n=args.num_candidates,
                bidirectional_generation=args.bidirectional_generation, bidirectional_scoring=args.bidirectional_scoring,
                truncation_parameters=truncation_parameters,
                scoring=args.candidate_scoring,
            )
            # if args.temperature == 0.0:
            #     # kwargs.update(sampling=False)
            # else:
            kwargs.update(sampling=True, top_p=args.top_p, temperature=args.temperature)
            sorted_choices, response = model.rank_infills(prompt_parts, **kwargs)

            top_choice = sorted_choices[0]

            results = problem.copy()
            
            assert len(prompt_parts) == 2
            results["text"] = top_choice["infills"][0]
            results["complete"] = top_choice["complete"]
            results["text_untruncated"] = top_choice["infills_untruncated"][0]

            responses[key] = response

            infill_results.append(results)

        result = {
                "task_id": task_id,
                "canonical_solution": problem["canonical_solution"],
                "infill_results": infill_results,
                }
        all_results.append(result)
        result_json.write(json.dumps(result) + "\n")
        if i % 10 == 0:
            result_json.flush()

    with open(result_pkl_fname, "wb") as f:
        pickle.dump(all_results, f)

    result_json.close()

def evaluate_systematic(filename: str, truncation_heuristics: List[str] = ["num_lines"]):
    """Reads infilled completions from a file, postprocesses them (truncates them to one line or multi-line w/ heuristics),
    and evaluates functional correctness (average pass rate / exact match).
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
                prefix, suffix = infill_res["prompt_parts"]

                truncation_parameters = TruncationParameters.from_heuristics(truncation_heuristics, infill_res["missing_lines"], suffix)

                infill_truncated = truncation_parameters.truncate(infill_res["text_untruncated"])

                # TODO: this strips initial whitespace. could check whether indent is correct?
                is_exact_match = infill_truncated.rstrip() == infill_res["missing_lines"].rstrip()

                complete = "".join([prefix, infill_truncated, suffix])

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
    # with open(f"{ext_stripped}__functional_eval.json", "w") as f:
    #     json.dump(functional_results, f)

if __name__ == "__main__":
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--tokenizer_name", type=str, choices=["gpt2", "gpt2_pretokenization_newlines_only"])
    parser.add_argument("--prompt_prefix", type=str)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--result_base_path")
    parser.add_argument("--eval_type", choices=["one_line", "all_lines"], default="one_line")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--truncation_heuristics", nargs='*', choices=TruncationParameters.HEURISTICS, default=["num_lines"])

    parser.add_argument("--bidirectional_generation", action="store_true")
    parser.add_argument("--bidirectional_scoring", action="store_true")

    # for LTR models
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--candidate_scoring", choices=["mean", "sum", "random"], default="mean")

    args = parser.parse_args()
    pprint.pprint(vars(args))

    if not args.evaluate_only:
        if args.model_path is None:
            # assert args.cached_responses, "must pass --model_path=<model> or --cached_responses"
            model = Model()
        else:
            model = make_model(args, args.model_path, args.tokenizer_name, prompt_prefix=args.prompt_prefix)
        run_systematic_infill(args, model, eval_type=args.eval_type, result_base_path=args.result_base_path)
    evaluate_systematic(f"{args.result_base_path}.pkl", truncation_heuristics=args.truncation_heuristics)