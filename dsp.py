from utils import dump_version_info, read_file, dump_git_status
import json
import pickle
import numpy as np
import tqdm
import os
import sys
import pprint
import glob

import argparse

from typing import List

from models import TruncationParameters, add_infilling_args, add_model_args, make_model, Model, FairseqModel

from data_science_problems.read import read_problems

def make_shard_string(shard_number, num_shards):
    if shard_number is None or shard_number < 0:
        return ""
    assert 0 <= shard_number < num_shards
    shard_string = f"_shard-{shard_number}-of-{num_shards}"
    return shard_string

def run_dsp(args, model: Model, result_base_path=None):
    """Masks out a subset of lines in the HumanEval reference solution and infills with the CM model. Saves the output to a file for later evaluation.

    Args:
        eval_type (str):
            "one_line": mask out all possible single lines
            "all_lines": mask out all possible multi-line chunks
    """

    incoder_format = isinstance(model, FairseqModel)

    print(f"incoder_format: {incoder_format}")

    if not incoder_format:
        raise NotImplementedError("need to define stop criterion for non-incoder models")

    problem_iterator = list(read_problems(incoder_format=incoder_format, context_len=args.context_len).items())

    if args.shard_number is not None and args.shard_number >= 0:
        shard_string = make_shard_string(args.shard_number, args.num_shards)
        shard_size = (len(problem_iterator) // args.num_shards) + 1
        shard_start = shard_size * args.shard_number
        shard_end = min(shard_start + shard_size, len(problem_iterator))
        print(f"sharding to only process instances [{shard_start}, {shard_end})")
        # problem_iterator = problem_iterator[shard_start:shard_end]
    else:
        shard_start = 0
        shard_end = len(problem_iterator)
        shard_string = ""

    all_results = []
    if result_base_path is not None:
        result_json_fname = f"{result_base_path}{shard_string}.json"
        result_pkl_fname = f"{result_base_path}{shard_string}.pkl"
        response_pkl_fname = f"{result_base_path}{shard_string}_responses.pkl"
    else:
        result_json_fname = f"dsp_{shard_string}.json"
        result_pkl_fname = f"dsp_{shard_string}.pkl"
        response_pkl_fname = f"dsp_{shard_string}.pkl"
    result_json = open(result_json_fname, "w")

    responses = {}

    functional_results = []
    infill_attempts = []

    def compute_metrics():
        metrics = {}
        return metrics

    indices = list(range(shard_start, shard_end))

    with tqdm.tqdm(indices, ncols=120) as pbar:
        for i in pbar:
            task_id, problem = problem_iterator[i]
            if functional_results:
                metrics = compute_metrics()
                pbar.set_postfix(metrics)

            prefix = problem["prompt"] + "<code>\n"
            suffix = "\n</code>"

            if args.include_tests:
                suffix += "\n" + problem["test"]

            prompt_parts = [prefix, suffix]

            # don't truncate
            truncation_parameters = [
                TruncationParameters.from_heuristics([], None, None)
            ]
            kwargs = dict(
                verbose=True, n=args.num_candidates,
                bidirectional_generation=args.bidirectional_generation, bidirectional_scoring=args.bidirectional_scoring,
                truncation_parameters=truncation_parameters,
                scoring=args.candidate_scoring,
                # TODO: figure out stop words for non-fairseq models (fairseq models include </code> etc by default)
                stop_words=[],
            )
            if args.max_tokens is not None:
                kwargs['max_tokens'] = args.max_tokens
            # if args.temperature == 0.0:
            #     # kwargs.update(sampling=False)
            # else:
            kwargs.update(sampling=True, top_p=args.top_p, temperature=args.temperature, beam=args.beam)
            sorted_choices, response = model.rank_infills(prompt_parts, **kwargs)

            top_choice = sorted_choices[0]
            if "infill_attempts" in top_choice:
                # infill_attempts is a list of len(parts) - 1, i.e. 1 for a single-infilling task
                infill_attempts.append(top_choice["infill_attempts"][0])

            result = problem.copy()
            
            assert len(prompt_parts) == 2
            result["completion"] = top_choice["infills"][0]
            result["complete"] = top_choice["complete"]
            result["completion_untruncated"] = top_choice["infills_untruncated"][0]

            responses[task_id] = response

            # infill_results.append(infill_result)

            # functional_results.append(eval_result(
            #     task_id, humaneval_problem, args.truncation_heuristics, infill_result,
            #     override_suffix=args.override_suffix,
            # ))

            with open(response_pkl_fname, "wb") as f:
                pickle.dump(responses, f)
            all_results.append(result)
            result_json.write(json.dumps(result) + "\n")
            if i % 10 == 0:
                result_json.flush()
    
    pprint.pprint(compute_metrics())

    with open(result_pkl_fname, "wb") as f:
        pickle.dump(all_results, f)

    result_json.close()

def make_parser():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_infilling_args(parser)

    parser.add_argument("--result_base_path")
    parser.add_argument("--include_tests", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--context_len", type=int, default=3)

    parser.add_argument("--git_status", action="store_true")

    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--shard_number", type=int, default=-1)


    return parser


if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    if not args.evaluate_only:
        model = make_model(args)
        run_dsp(args, model, result_base_path=args.result_base_path)
    # shard_string = make_shard_string(args.shard_number, args.num_shards)
    # evaluate_systematic(args)