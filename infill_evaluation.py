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

from models import make_model, Model

from he import HUMAN_EVAL_STOP_WORDS, generate_he_infill_problems

def run_systematic_infill(args, model: Model, eval_type="one_line", result_base_path=None):
    """Masks out a subset of lines in the HumanEval reference solution and infills with the CM model. Saves the output to a file for later evaluation.

    Args:
        eval_type (str):
            "one_line": mask out all possible single lines
            "all_lines": mask out all possible multi-line chunks
    """

    assert eval_type in ("one_line", "all_lines")

    results = []
    if result_base_path is not None:
        result_json_fname = f"{result_base_path}.json"
        result_pkl_fname = f"{result_base_path}.pkl"
    else:
        result_json_fname = f"he_{eval_type}_systematic.json"
        result_pkl_fname = f"he_{eval_type}_systematic.pkl"
    result_json = open(result_json_fname, "w")

    problem_iterator = generate_he_infill_problems(args, eval_type)

    for i, (task_id, task_id_problems) in enumerate(tqdm.tqdm(problem_iterator, ncols=120)): 
        
        infill_results = []
        for problem in task_id_problems:
            prompt_parts = problem["prompt_parts"]
            if args.temperature == 0.0:
                out = model.infill(prompt_parts, verbose=False, sampling=False)
            else:
                out = model.infill(prompt_parts, verbose=False, sampling=True, sampling_topp=args.top_p, sampling_temperature=args.temperature)

            if len(out["infills"]) != 1:
                raise NotImplementedError("multiple region infilling is not implemented")
            infill = out["infills"][0]

            complete = ''.join(out["complete"])

            results = problem.copy()
            results["infill"] = infill
            results["complete"] = complete

            infill_results.append(results)
            # print("="*20)
            # print("Prompt: ", prompt_parts)
            # print(out["infills"][0])

        result = {
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

def evaluate_systematic(filename: str, truncation_heuristic: str = "num_lines", rerank_with_right_context: bool = False):
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
                left_context, right_context = infill_res["prompt_parts"]
                if rerank_with_right_context:
                    # Need to postprocess all candidates (and then re-rank)
                    candidate_infills = infill_res["infill_candidates"]
                else:
                    if "infill_candidates" in infill_res:
                        _, top_candidate_text = infill_res["infill_candidates"][0]
                        candidate_infills = [top_candidate_text]
                    else:
                        candidate_infills = [infill_res["infill"]]

                candidate_infills_truncated = []
                for infill in candidate_infills:
                    missing_lines = stripped_line_split(infill_res["missing_lines"])
                    if truncation_heuristic == "num_lines":
                        #max_num_lines = max(1, infill_res["missing_lines"].count("\n")
                        infill_truncated = truncate_num_lines(infill, max_num_lines=len(missing_lines))
                    elif truncation_heuristic == "stopwords":
                        raise NotImplementedError()
                    elif truncation_heuristic == "suffix":
                        infill_truncated = truncate_overlap(infill, right_context, num_consecutive_lines=4)
                    elif truncation_heuristic == "suffix+num_lines":
                        infill_truncated = truncate_overlap(infill, right_context, num_consecutive_lines=4)
                        infill_truncated = truncate_num_lines(infill_truncated, max_num_lines=len(missing_lines))
                    else:
                        raise NotImplementedError()
                    candidate_infills_truncated.append(infill_truncated)

                if rerank_with_right_context:
                    infill_truncated = rerank_with_right_context(candidate_infills, left_context, right_context)
                else:
                    infill_truncated = candidate_infills_truncated[0]

                # TODO: this strips initial whitespace. could check whether indent is correct?
                is_exact_match = infill_truncated.rstrip() == infill_res["missing_lines"].rstrip()

                complete = "".join([left_context, infill_truncated, right_context])

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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_name", type=str, choices=["gpt2", "gpt2_pretokenization_newlines_only"])
    parser.add_argument("--prompt_prefix", type=str)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--result_base_path")
    parser.add_argument("--eval_type", choices=["one_line", "all_lines"], default="one_line")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--truncation_heuristic", choices=["num_lines", "suffix", "suffix+num_lines"], default="num_lines")

    # for LTR models
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--candidate_scoring", choices=["mean", "sum"], default="mean")

    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.model_name is None:
        assert args.cached_responses, "must pass --model_path=<model> or --cached_responses"
        model = Model()
    else:
        model = make_model(args, args.model_path, args.tokenizer_name, prompt_prefix=args.prompt_prefix)

    if not args.evaluate_only:
        run_systematic_infill(args, model, eval_type=args.eval_type, result_base_path=args.result_base_path)
    evaluate_systematic(f"{args.result_base_path}.pkl", truncation_heuristic=args.truncation_heuristic)
