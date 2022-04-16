import json
import sys
import pprint
import tqdm
import pprint
from collections import Counter
import pprint

from datasets import load_dataset

from models import add_model_args, add_infilling_args, make_model

from type_hints import create_return_example
from typewriter import load_json, load_pickle, run_return_prediction, force_none_for_no_returns, evaluate

from utils import dump_git_status, dump_version_info

def build_examples(split: str = 'validation', show_tqdm=False):
    data = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)

    problems = [{
        "id": d["id"],
        "original_string": d["original_string"],
    } for d in data]

    return_examples = []
    examples_found = 0

    it = problems
    if show_tqdm:
        it = tqdm.tqdm(it, ncols=120)
    for ix, problem in enumerate(it):
        ex = create_return_example(problem["original_string"], None, None, False)
        if show_tqdm:
            it.set_postfix({"examples": f"{examples_found} / {ix+1} ({examples_found*100/(ix+1):.2f}%)"})
        if ex is not None:
            return_examples.append(ex)
            examples_found += 1
    
    print(f"return: found examples for {examples_found} / {len(problems)} ({examples_found/len(problems)*100:.2f}%)")
    return return_examples

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()

    add_model_args(parser)
    add_infilling_args(parser)

    parser.add_argument("--example_base_filename", default="data/codexglue_return_types")
    parser.add_argument("--git_status", action="store_true")
    parser.add_argument("--generate_examples", action="store_true")
    parser.add_argument("--result_base_path")
    parser.add_argument("--num_examples", type=int)

    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--shard_number", type=int, default=-1)

    parser.add_argument("--serialized_results_paths", nargs='*')

    parser.add_argument("--force_none_for_no_returns", action="store_true")
    parser.add_argument("--skip_none_type", action="store_true")

    parser.add_argument("--split", choices=["validation", "test"], default="validation")

    return parser

if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    split = args.split

    filename = f"{args.example_base_filename}_{split}.json"

    if args.generate_examples:
        examples = build_examples(split, show_tqdm=True)
        with open(filename, 'w') as f:
            json.dump(examples, f, indent=4)
    else:
        examples = load_json(filename)
        if args.num_examples:
            examples = examples[:args.num_examples]

        if args.serialized_results_paths is not None:
            results = []
            for path in args.serialized_results_paths:
                results += load_pickle(path)
        else:
            model = make_model(args)
            # 10 for some buffer
            model.max_seq_length = 2048-10

            results = run_return_prediction(args, examples, model, result_base_path=args.result_base_path, taskname="codexglue_return_types")
        
        if args.force_none_for_no_returns:
            results = [force_none_for_no_returns(result) for result in results]

        if args.skip_none_type:
            results = [result for result in results if result["return_type_from_source"] != "None"]
        
        if len(results) != len(examples):
            print(f"warning: {len(examples)} examples but {len(results)} results")
        pprint.pprint(evaluate(results, type_from_source=True))
