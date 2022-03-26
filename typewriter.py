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

from type_hints import create_return_example

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
            'predicted_types': predicted
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

def build_examples(typewriter_dir: str, crawl_root: str, decontaminate: bool, imports_and_function_only: bool, split: str = 'validation', show_tqdm=False):
    repo_to_commit, repo_to_name, repo_to_paths = read_typewriter_data(typewriter_dir, split=split)

    repos = set(repo_to_paths.keys())

    if decontaminate:
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
        predicted_types = record['predicted_types']
        if repo not in repos:
            skip_reasons['no repo'] += 1
            continue
        archive = repo_to_archive[repo]
        matching_files = [f for f in archive['file_data'] if f['path'] == path]
        if len(matching_files) != 1:
            print(f"found {len(matching_files)} for {repo}, {path}")
            skip_reasons[f"file mismatch {len(matching_files)}"] += 1
            continue
        source = matching_files[0]['content']
        ex = create_return_example(source, lineno, true_type, imports_and_function_only)
        if ex is not None:
            return_examples.append(ex)
        else:
            skip_reasons["no ast match"] += 1
            continue
    
    pprint.pprint(skip_reasons)
    print(f"return: skipped {sum(skip_reasons.values())} / {len(result_predictions)} examples ({len(return_examples)} remaining)")
    return return_examples

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--typewriter_dir", default="/private/home/dpf/data/TypeWriter_dataset")
    parser.add_argument("--crawl_root", default="/checkpoint/dpf/data/typewriter/crawl/data")
    parser.add_argument("--git_status", action="store_true")
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
    decontaminate = True
    imports_and_function_only = True

    examples = build_examples(args.typewriter_dir, args.crawl_root, decontaminate, imports_and_function_only, split)