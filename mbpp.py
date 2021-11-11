# author: Freda Shi
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
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils import Command
from functools import partial

from models import complete_codex_persistent

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

def evaluate_code_generic(prompt_completion_function, output_path, num_attempts=1, verbose=False):
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
            for batch_ix in range(num_attempts):
                text = item['text']
                test_setups = item['test_setup_code']
                test_cases = item['test_list']
                new_function_name = None
                prompt = generate_prompt(text, test_cases[0], new_function_name)
                all_code = prompt_completion_function(prompt)

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

def codex_greedy(prompt: str, max_tokens=450):
    response = complete_codex_persistent(
        engine="davinci-codex",
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=MBPP_STOP_WORDS,
    )
    return [response['choices'][0]['text']]

def codex_sample(prompt: str, max_tokens=450, top_p=0.95, num_samples=15, temperature=0.5):
    response = complete_codex_persistent(
        engine="davinci-codex",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop=MBPP_STOP_WORDS,
        n=num_samples,
    )
    return [choice['text'] for choice in response['choices']]

def gpt2_greedy(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    if inputs.input_ids.shape[1] > model.config.max_length:
        inputs = inputs[:,:model.config.max_length]
    generation_output = model.generate(**inputs, return_dict_in_generate=True)
    code = tokenizer.decode(generation_output['sequences'].tolist()[0])
    code = code.replace('<|endoftext|>', '')
    return [code]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-path', type=str, default='../../models/pretrained/codemodels/code-gpt2/pytorch_model.bin')
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--eval-mode', type=str, default='greedy', choices=['greedy', 'sample'])
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--num-attempts', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    #parser.add_argument('--func-name-length', type=int, default=30)
    args = parser.parse_args()

    random.seed(1)

    dataset = MBPPDataset()
    if args.model == 'code-gpt2':
        # load models
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        model.config.max_length = model.config.n_positions 
        model.config.pad_token_id = model.config.eos_token_id
        if args.model_path:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
        # parallellize for multi-GPU training
        n_devices = torch.cuda.device_count()
        layers_per_device = model.config.n_layer // n_devices + 1
        device_map = {k: [i for i in range(layers_per_device * k, min(layers_per_device * (k+1), model.config.n_layer))] for k in range(n_devices)}
        model.parallelize(device_map)

        assert args.eval_mode == 'greedy'
        completion_function = partial(gpt2_greedy, model=model, tokenizer=tokenizer)
    elif args.model == 'codex':
        completion_function = codex_greedy
    elif args.model == 'codex-sample':
        completion_function = codex_sample
    elif args.model == 'codex-postproc':
        raise NotImplementedError()
    else:
        raise Exception('Model not supported yet')
    evaluate_code_generic(completion_function, args.output_path, num_attempts=args.num_attempts, verbose=args.verbose)
