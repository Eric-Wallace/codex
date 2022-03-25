import re
import json
import os

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

def read_prediction_file(path, argument=False):
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

def read_typewriter_validation(typewriter_dir):
    repo_to_commit, repo_to_name = read_github_url_file(os.path.join(typewriter_dir, 'github_urls.txt'))
    # Dict[str, List[str]]: name -> paths
    repo_to_paths = read_filename_file(os.path.join(typewriter_dir, 'open_source_validation_files.txt'))
    return repo_to_commit, repo_to_name, repo_to_paths

if __name__ == "__main__":
    pass