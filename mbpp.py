# modified from code by Freda Shi
import argparse
import os
import random
import json
import tempfile
import torch
import os
import signal
import subprocess
import threading
from tqdm import tqdm
from functools import partial

from models import make_model

# stop words differ from human_eval because we don't have the function signature
# TODO: consider standardizing these interfaces
MBPP_STOP_WORDS = ["\nassert", "\nclass", "\nif", "\nprint"]

class MBPPDataset(object):
    def __init__(self, path='/private/home/fhs/data/mbpp/mbpp.jsonl'):
        self.data = [json.loads(line) for line in open(path)]

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode

def generate_prompt(description, test_example):
    return f'"""\n{description}\n{test_example}\n"""'

def evaluate_code_generic(model, output_path, verbose=False, num_candidates_generated=15, num_candidates_evaluated=15):
    """
    prompt_completion_function should be prompt function of the sort: lambda prompt: str -> {
        ...,
        'choices': [{
            ...,
            'text': completion: str,
            ...,
        }],
        ...
    }
    """
    tempdir = tempfile.TemporaryDirectory(prefix='mbpp-eval')
    # this will be used temporarily for all solutions
    code_path = f'{tempdir.name}/code.py'
    problems_passed = 0
    total_problems = 0
    successes = 0
    attempts = 0
    try:
        os.system(f'mkdir -p {output_path}')
        bar = tqdm(dataset.data, ncols=80)
        for i, item in enumerate(bar):
            this_attempts = 0
            some_attempt_passed = False
            text = item['text']
            test_setups = item['test_setup_code']
            test_cases = item['test_list']
            new_function_name = None
            prompt = generate_prompt(text, test_cases[0], new_function_name)

            if num_candidates_evaluated != num_candidates_generated:
                raise NotImplementedError()
            response = model.complete(prompt, MBPP_STOP_WORDS, n=num_candidates_generated)
            all_code = [choice['text'] for choice in response['choices']]
            for code_ix, code in enumerate(all_code):
                this_attempts += 1
                if verbose:
                    print(f"PROBLEM {i} ATTEMPT {this_attempts}")
                solution_path = f'{output_path}/solution-{i:03d}-{this_attempts}.py'
                # if os.path.exists(solution_path):  # fix bug: if prompts don't match, rerun
                #     existing_prompt = ''.join(open(solution_path).readlines()[:5]).strip()
                #     if existing_prompt == prompt:
                #         continue

                if verbose:
                    print("<PROMPT>")
                    print(prompt)
                    print("</PROMPT>")

                # write code to file 
                with open(code_path, 'w') as fout:
                    print(prompt, file=fout)
                    print(code, file=fout)
                    if verbose:
                        print("<COMPLETION>")
                        print(code)
                        print("</COMPLETION>")
                    print(test_setups, file=fout)
                    if verbose:
                        print("<TESTS>")
                        print(test_setups)
                    for case in test_cases:
                        print(case, file=fout)
                        if verbose:
                            print(case)
                    if verbose:
                        print("</TESTS>")
                    fout.close()
                os.system(f'cp {code_path} {solution_path}')
                command = Command(f'python {code_path} >/dev/null 2>&1')
                this_passed = (command.run(timeout=args.timeout) == 0)
                successes += this_passed
                if verbose:
                    print("PASSED" if this_passed else "FAILED")
                    print()
                attempts += 1
                some_attempt_passed |= this_passed
                #bar.set_description(f'{successes}/{attempts} successes.')
            if some_attempt_passed:
                problems_passed += 1
            total_problems += 1
            bar.set_description(f'ps: {problems_passed / total_problems * 100:.1f}%; as: {successes/attempts*100:.1f}%')
        tempdir.cleanup()
    except KeyboardInterrupt:
        tempdir.cleanup()
    print(f'problem success: {problems_passed}/{total_problems} passed ({problems_passed/total_problems*100:.2f}%): ')
    print(f'attempt success: {successes}/{attempts} ({successes/attempts*100:.2f}%); ({attempts / total_problems:.2f} per-problem)')
    return problems_passed / total_problems

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None, required=False)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument("--num_candidates_generated", type=int, default=15)
    parser.add_argument("--num_candidates_evaluated", type=int, default=15)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    random.seed(1)

    dataset = MBPPDataset()

    model = make_model(args.model_name, args.tokenizer_name)
    evaluate_code_generic(model, args.output_path, verbose=args.verbose, num_candidates_generated=args.num_candidates_generated, num_candidates_evaluated=args.num_candidates_evaluated)