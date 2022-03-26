import re
import json
import os
import sys
import pprint
import csv
import glob
import tqdm
from typing import Set
import pprint
from collections import Counter
import pickle
import pprint

from models import Model, TruncationParameters, add_model_args, add_infilling_args, make_model

from type_hints import create_return_example, normalize_type

import zstandard as zstd

from utils import dump_git_status, dump_version_info

UNK = '@@UNK@@'

line_url_re = re.compile(r'https://github.com/(.*)/(.*)\.git::(.*)')
arg_re = re.compile(r'(\d+)->\[(.*)\]')

def read_github_url_file(path):
    repo_to_commit = {}
    repo_to_name = {}
    with open(path, 'r') as f:
        for line in f:
            match = line_url_re.match(line.strip())
            user, repo, commit = match.groups()
            assert repo not in repo_to_commit
            repo_to_commit[repo] = commit
            repo_to_name[repo] = f'{user}/{repo}'
    return repo_to_commit, repo_to_name

def read_filename_file(path):
    names_to_paths = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            split = line.split('/')
            repo, *rest = split
            name = repo
            if name not in names_to_paths:
                names_to_paths[name] = []
            names_to_paths[name].append('/'.join(rest))
    return names_to_paths

def read_prediction_file(typewriter_dir, argument=False):
    if argument:
        path = os.path.join(typewriter_dir, "open_source_results_argument.json")
    else:
        path = os.path.join(typewriter_dir, "open_source_results_return.json")
    prefix = "/home/anon/local/github-repos/"
    with open(path, 'r') as f:
        data = json.load(f)
    records = []
    for key, answers in data.items():
        key = key.strip()
        file, location = key.split(' : ')
        if argument:
            line_num, var = arg_re.match(location).groups()
            line_num = int(line_num)
        else:
            line_num = int(location)
        assert file.startswith(prefix)
        repo, *rest = file.lstrip(prefix).split('/')
        path = '/'.join(rest)
        gold = answers[0]
        if gold == UNK:
            gold = None
        # predicted = [t if t != UNK else None for t in answers[1]]
        predicted = answers[1]
        record = {
            'repo': repo,
            'path': path,
            'line_number': line_num,
            'true_type': gold,
            'typewriter_predicted_types': predicted
        }
        if argument:
            record['variable'] = var
        records.append(record)
    return records

def read_typewriter_data(typewriter_dir, split='validation'):
    repo_to_commit, repo_to_name = read_github_url_file(os.path.join(typewriter_dir, 'github_urls.txt'))
    # Dict[str, List[str]]: name -> paths
    repo_to_paths = read_filename_file(os.path.join(typewriter_dir, f'open_source_{split}_files.txt'))
    return repo_to_commit, repo_to_name, repo_to_paths

def remove_crawled_repos(repos: Set[str]):
    """" filter down repos to exclude ones that we trained on"""
    crawled_repos = set()
    for fname in glob.glob('/checkpoint/dpf/data/github/*.csv'):
        with open(fname, 'r') as f:
            for record in csv.DictReader(f):
                if 'name' in record:
                    key = 'name'
                else:
                    key = 'repo_name'
                    assert key in record
                crawled_repos.add(record[key])
    print(f"before removing crawled repos: {len(repos)}")
    decon = set(repos) - set(crawled_repos)
    print(f"after removing crawled repos: {len(decon)}")
    return decon

def read_source_files(crawl_root, repos: Set[str], repo_to_paths, repo_to_name):
    compressor = zstd.ZstdDecompressor()
    repo_to_archive = {}
    skipped = []
    for repo in repos:
        paths = repo_to_paths[repo]
        name = repo_to_name[repo]
        archive_fname = os.path.join(crawl_root, f"{name}.json.zstd")
        if not os.path.exists(archive_fname):
            print(f"archive {archive_fname} not found; removing from repos")
            skipped.append(repo)
            continue
        with open(archive_fname, 'rb') as f:
            reader = compressor.stream_reader(f)
            archive = json.load(reader)
            repo_to_archive[repo] = archive
    for repo in skipped:
        repos.remove(repo)
    return repo_to_archive

def build_examples(typewriter_dir: str, crawl_root: str, imports_and_function_only: bool, split: str = 'validation', show_tqdm=False):
    repo_to_commit, repo_to_name, repo_to_paths = read_typewriter_data(typewriter_dir, split=split)

    repos = set(repo_to_paths.keys())
    repos = remove_crawled_repos(repos)

    repo_to_archive = read_source_files(crawl_root, repos, repo_to_paths, repo_to_name)

    result_predictions = read_prediction_file(typewriter_dir, argument=False)
    argument_predictions = read_prediction_file(typewriter_dir, argument=True)

    return_examples = []

    skip_reasons = Counter()

    it = result_predictions
    if show_tqdm:
        it = tqdm.tqdm(it, ncols=120)
    for record in it:
        repo = record['repo']
        path = record['path']
        lineno = record['line_number']
        true_type = record['true_type']
        predicted_types = record['typewriter_predicted_types']
        if repo not in repos:
            skip_reasons['no repo'] += 1
            continue
        archive = repo_to_archive[repo]
        matching_files = [f for f in archive['file_data'] if f['path'] == path]
        if len(matching_files) != 1:
            # print(f"found {len(matching_files)} for {repo}, {path}")
            skip_reasons[f"file mismatch {len(matching_files)}"] += 1
            continue
        source = matching_files[0]['content']
        ex = create_return_example(source, lineno, true_type, imports_and_function_only)
        if ex is not None:
            ex['true_type'] = record['true_type']
            if ex['true_type'] != None and normalize_type(ex['true_type']) != normalize_type(ex['return_type_from_source']):
                raise Exception(f"type mismatch: {normalize_type(ex['true_type'])} != {normalize_type(ex['return_type_from_source'])}")
            ex['typewriter_predicted_types'] = predicted_types
            return_examples.append(ex)
        else:
            skip_reasons["no ast match"] += 1
            continue
    
    pprint.pprint(skip_reasons)
    print(f"return: skipped {sum(skip_reasons.values())} / {len(result_predictions)} examples ({len(return_examples)} remaining)")
    return return_examples

def run_return_prediction(args, examples, model: Model, result_base_path=None):
    all_results = []
    responses = {}

    if result_base_path is not None:
        result_pkl_fname = f"{result_base_path}.pkl"
        response_pkl_fname = f"{result_base_path}_responses.pkl"
    else:
        result_pkl_fname = f"typewriter.pkl"
        response_pkl_fname = f"typewriter.pkl"

    with tqdm.tqdm(examples, ncols=120) as pbar:
        for i, problem in enumerate(pbar):
            # TODO: could add extra sentinels here to take the place of omitted code, 
            # if we're doing bidirectional_generation with our CM model
            left = "\n".join(problem["extra_left"]) + problem["left"]
            right = problem["right"]
            prompt_parts = [left, right]
            stop_words = [':']
            truncation_parameters = [
                    TruncationParameters.from_heuristics(["stop_words"], stop_words=stop_words)
                ]
            kwargs = dict(
                verbose=False, n=args.num_candidates,
                bidirectional_generation=args.bidirectional_generation, bidirectional_scoring=args.bidirectional_scoring,
                truncation_parameters=truncation_parameters,
                scoring=args.candidate_scoring,
                stop_words=stop_words,
            )
            if args.max_tokens is not None:
                kwargs['max_tokens'] = args.max_tokens
            kwargs.update(sampling=True, top_p=args.top_p, temperature=args.temperature, beam=args.beam)
            sorted_choices, response = model.rank_infills(prompt_parts, **kwargs)
            responses[i] = response
            top_choice = sorted_choices[0]

            infill_result = problem.copy()
            infill_result["predicted_type"] = top_choice["infills"][0]
            infill_result["complete"] = top_choice["complete"]
            infill_result["prediction_untruncated"] = top_choice["infills_untruncated"][0]

            verbose_output = f"{problem['true_type']} : {infill_result['predicted_type']}"

            pbar.set_postfix({"output": verbose_output})

            all_results.append(infill_result)

    # Note: since we don't save until the end of the run, these results are
    #  incomplete if we're resuming
    with open(result_pkl_fname, "wb") as f:
        pickle.dump(all_results, f)

    with open(response_pkl_fname, "wb") as f:
        pickle.dump(responses, f)

    return all_results

def get_typewriter_predictions(examples):
    all_results = []
    for problem in examples:
        result = problem.copy()
        # the first type in the list is the highest-confidence one
        result["predicted_type"] = result["typewriter_predicted_types"][0]
        all_results.append(result)
    return all_results

def evaluate(results, verbose=False, type_from_source=True):
    # pass type_from_source = False if running on all predictions from Typewriter (which include files for which we don't have source)
    # the method should give the same results for type_from_source=False and type_from_source=True (if it doesn't throw an error for missing source, with type_from_source=True)
    n_correct = 0
    n_predictions = 0
    n_types = 0
    for ix, result in enumerate(results):
        n_types += 1
        if type_from_source:
            true_type = result["return_type_from_source"]
        else:
            true_type = result["true_type"]
        if true_type is not None:
            true_type = normalize_type(true_type)
        if verbose:
            print(f"** example {ix} **")
            print("left:")
            print(result["left"])
            print("right:")
            print(result["right"][:10])
            print("true type:")
            print(true_type)
        if result["predicted_type"] == UNK:
            pred_type = None
        else:
            pred_type = normalize_type(result["predicted_type"])
        this_correct = pred_type == true_type
        if pred_type is not None:
            n_predictions += 1
            if this_correct:
                n_correct += 1
        if verbose:
            print("pred type:")
            print(pred_type)
            print(f"correct?\t{this_correct}")
            print()
    precision = n_correct / n_predictions
    recall = n_correct / n_types
    f1 = 2 * precision * recall / (precision + recall)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_correct': n_correct,
        'n_predictions': n_predictions,
        'n_types': n_types
    }

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()

    add_model_args(parser)
    add_infilling_args(parser)

    parser.add_argument("example_filename")
    parser.add_argument("--typewriter_dir", default="/private/home/dpf/data/TypeWriter_dataset")
    parser.add_argument("--crawl_root", default="/checkpoint/dpf/data/typewriter/crawl/data")
    parser.add_argument("--git_status", action="store_true")
    parser.add_argument("--generate_examples", action="store_true")
    parser.add_argument("--result_base_path")
    parser.add_argument("--num_examples", type=int)
    return parser

if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    split = 'validation'
    imports_and_function_only = True

    if args.generate_examples:
        examples = build_examples(args.typewriter_dir, args.crawl_root, imports_and_function_only, split, show_tqdm=True)
        with open(args.example_filename, 'w') as f:
            json.dump(examples, f, indent=4)
    else:
        with open(args.example_filename, 'r') as f:
            examples = json.load(f)
        
        if args.num_examples:
            examples = examples[:args.num_examples]
        model = make_model(args)

        results = run_return_prediction(args, examples, model, result_base_path=args.result_base_path)
        pprint.pprint(evaluate(results, type_from_source=True))