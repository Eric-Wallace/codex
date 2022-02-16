import sys
import json
import time
import tqdm
import pickle
import pprint
import argparse
from collections import defaultdict

from human_eval.data import write_jsonl, read_problems

from models import make_model, Model

# can't include print since the codex API will only handle up to 4 stop words
HUMAN_EVAL_STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif"]

def all_equal(iterable):
    iterable = list(iterable)
    return all(iterable[0] == x for x in iterable)

def unpickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def combine_responses(list_of_responses):
    assert all_equal(resp.keys() for resp in list_of_responses)
    responses = {}
    for resp in list_of_responses:
        for problem_id in resp:
            if problem_id not in responses:
                responses[problem_id] = {'choices': []}
            responses[problem_id]['choices'].extend(resp[problem_id]['choices'])
    return responses

if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer_name", type=str, choices=["gpt2", "gpt2_pretokenization_newlines_only"])
    parser.add_argument("--num_problems", type=int)
    parser.add_argument("--num_candidates_generated", type=int, default=15)
    parser.add_argument("--num_candidates_evaluated", type=int, default=1)
    parser.add_argument("--output_filename", default="samples.jsonl")
    parser.add_argument("--response_filename", default="responses.pkl")
    parser.add_argument("--cached_responses", action='store_true')
    parser.add_argument("--multiple_cached_responses_filenames", nargs='*')
    parser.add_argument("--remove_test_cases", default=False, action='store_true')

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--prompt_prefix")
    parser.add_argument("--candidate_scoring", choices=["mean", "sum", "random"], default="mean")

    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.model_name is None:
        assert args.cached_responses, "must pass --model_name=<model> or --cached_responses"
        model = Model()
    else:
        model = make_model(args, args.model_name, args.tokenizer_name, prompt_prefix=args.prompt_prefix)

    problems = list(sorted(read_problems().items()))
    if args.num_problems is not None:
        problems = problems[:args.num_problems]

    samples_to_evaluate = []
    if args.cached_responses:
        if args.multiple_cached_responses_filenames:
            responses = combine_responses([unpickle(fname) for fname in args.multiple_cached_responses_filenames])
        else:
            responses = unpickle(args.response_filename)
    else:
        responses = {}
    for task_id, problem in tqdm.tqdm(problems, ncols=80):
        prompt = problem['prompt']
        if args.remove_test_cases:
            if 'double_the_difference([' in prompt:
                prompt = prompt.split("double_the_difference([")[0].strip() + '\n    """'
            if '[input/output] samples' in prompt:
                prompt = prompt.split("[input/output] samples")[0].strip() + '\n    """'
            if 'compare_one' in prompt:
                prompt = prompt.split("compare_one")[0].strip() + '\n    """'
            if 'fix_spaces"' in prompt:
                prompt = prompt.split('fix_spaces"')[0].strip() + '\n    """'
            if 'It must be implemented like this:' in prompt:
                prompt = prompt.split("It must be implemented like this:")[0].strip() + '\n    """'
            if 'next_smallest([' in prompt:
                prompt = prompt.split("next_smallest([")[0].strip() + '\n    """'
            elif "is_nested('[[]]')" in prompt:
                prompt = prompt.split("is_nested('[[]]')")[0].strip() + '\n    """'
            elif 'for example' in prompt:
                prompt = prompt.split('for example')[0].strip() + '\n    """'
            elif 'For Example' in prompt:
                prompt = prompt.split('For Example')[0].strip() + '\n    """'
            elif 'Examples' in prompt:
                prompt = prompt.split('Examples')[0].strip() + '\n    """'
            elif 'Example' in prompt:
                prompt = prompt.split('Example')[0].strip() + '\n    """'
            elif 'For example' in prompt:
                prompt = prompt.split('For example')[0].strip() + '\n    """'
            elif '>>>' in prompt:
                prompt = prompt.split('>>>')[0].strip() + '\n    """'
        completions, response = model.rank_completions(
            prompt, HUMAN_EVAL_STOP_WORDS,
            max_tokens=450,
            n=args.num_candidates_generated,
            # if we've cached responses, use the cached
            cached_response=responses.get(task_id) if args.cached_responses else None,
            scoring=args.candidate_scoring,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        responses[task_id] = response
        for score, candidate in completions[:args.num_candidates_evaluated]:
            samples_to_evaluate.append(dict(
                task_id=task_id,
                completion=candidate
            ))

    write_jsonl(args.output_filename, samples_to_evaluate)
    with open(args.response_filename, 'wb') as f:
        pickle.dump(responses, f)

    from human_eval.evaluation import evaluate_functional_correctness
    import pprint
    results, extra = evaluate_functional_correctness(sample_file=None, samples=samples_to_evaluate)
    pprint.pprint(results)
