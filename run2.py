import argparse
import numpy as np
import openai
import json
import time
import tqdm
import pickle
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq.hub_utils import from_pretrained, GeneratorHubInterface
import torch

lm_model = None
lm_tokenizer = None
stop_words = None
def setup_model(model_name, tokenizer_name=None):
    if 'davinci' in model_name or 'cushman' in model_name: # openai requires no setup
        return

    # load the GPT-2 model
    global lm_model
    global lm_tokenizer
    global stop_words
    if lm_model is None:
        if 'fairseq' in model_name:
            #fairseq_pretrain = from_pretrained('/'.join(model_name.split('/')[0:-1]), 'checkpoint.pt', bpe='gpt2')
            fairseq_pretrain = from_pretrained(model_name, 'model.pt')
            lm_model = GeneratorHubInterface(fairseq_pretrain['args'], models=fairseq_pretrain['models'], task=fairseq_pretrain['task'] )
            lm_model.eval().cuda() # TODO do half()
            stop_words = [lm_model.encode(token)[0:1].tolist() for token in ["classdasdas", "def", "#", "if", "print"]] # cut off eos with -1
        else:
            lm_model = AutoModelForCausalLM.from_pretrained(model_name)
            lm_model.eval().cuda() # TODO do half()
        
            if tokenizer_name == None:
                tokenizer_name = model_name
            lm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            stop_words = [lm_tokenizer.encode(token) for token in ["class", "def", "#", "if", "print"]]

def complete_local_model(prompt, max_tokens=450, top_p=0.95, n=1, num_log_probs=1):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    assert isinstance(prompt, str)
    prompt = [prompt] # below code assumes list

    input_ids = lm_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=False)
    choices = []
    while len(choices) < n:
        num_to_sample = min(5, n - len(choices))
        with torch.inference_mode():
            # generate from the model
            total_sequences = lm_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=max_tokens + len(input_ids['input_ids'][0]), do_sample=True, num_return_sequences=n, top_p=top_p, early_stopping=True, use_cache=True, temperature=0.6)

            # now do something dumb where you run the model another time to get the probs
            logits = lm_model.forward(input_ids=total_sequences, return_dict=True).logits.detach().cpu()
            # get the top tokens and probs for the generated tokens
            probs = torch.softmax(logits[:,-max_tokens-1:], dim=2).cpu()
            top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
            logprobs = torch.log(probs)
            top_log_probs = torch.log(top_probs)

        # create the return value to resemble OpenAI
        return_json = {}
        for batch_id in range(num_to_sample):
            curr_json = {}

            # if you we find one of the stopwords, then we delete everything from the stopword on
            curr_max_tokens = None
            for stop_word_tensor in stop_words:
                if stop_word_tensor[0] in total_sequences[batch_id][-max_tokens:].tolist(): # if stopword is in the list
                    possible_stop_index = total_sequences[batch_id][-max_tokens:].tolist().index(stop_word_tensor[0]) # get the position of stopword
                    if possible_stop_index != -1 and total_sequences[batch_id][-max_tokens:].tolist()[possible_stop_index - 1] == 198: # 198 is the \n character. Assert that its in the position before the stopword
                        if curr_max_tokens is None or possible_stop_index < curr_max_tokens: # save first occruene of stopword
                            curr_max_tokens = possible_stop_index - 1 # -1 to also cut off \n

            if curr_max_tokens is not None: # stopword is found, cut stuff off
                curr_json['text'] = lm_tokenizer.decode(total_sequences[batch_id][-max_tokens:-max_tokens + curr_max_tokens], skip_special_tokens=True)
            else:
                print('no stopword found!') # not having the stopword found is probably a very bad sign
                curr_json['text'] = lm_tokenizer.decode(total_sequences[batch_id][-max_tokens:], skip_special_tokens=True)


            # fill the return json with the top tokens and probs to match the OpenAI return value.
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            # cutoff the -1 here because the probs are shifted one over for LMs
            if curr_max_tokens is None: # no stopword
                curr_top_log_probs = top_log_probs[batch_id][:-1]
                curr_top_tokens = top_tokens[batch_id][:-1]
            else:
                curr_top_log_probs = top_log_probs[batch_id][:curr_max_tokens-1]
                curr_top_tokens = top_tokens[batch_id][:curr_max_tokens-1]
            for current_element_top_log_probs, current_element_top_tokens in zip(curr_top_log_probs, curr_top_tokens):
                # tokens is a list of the top token at each position
                curr_json['logprobs']['tokens'].append(lm_tokenizer.decode([current_element_top_tokens[0]]))
                # token_logprobs is a list of the logprob of the top token at each position
                curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                temp = {}
                for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                    temp[lm_tokenizer.decode(token.item())] = log_prob.item()
                curr_json['logprobs']['top_logprobs'].append(temp)

            choices.append(curr_json)
    return_json['choices'] = choices
    return return_json

def complete_fairseq_lm(prompt, max_tokens=450, top_p=0.95, n=1, num_log_probs=1):
    ''' This function runs fairseq LM locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    # need the topk score and token at each step
    from fairseq.data.data_utils import collate_tokens
    from fairseq import utils

    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list

    pad_idx, eos_idx, bos_idx = lm_model.src_dict.pad(), lm_model.src_dict.eos(), lm_model.src_dict.bos()

    # greedily generate l tokens
    # the generate function can handle left padded inputs automatically in HF
    # total_sequences is now the input + possible generated output
    choices = []
    with torch.inference_mode():
        # TODO, for some reason I can't get the sample() to actually return more than one candidaete
        lm_model.args.batch_size=n
        lm_model.args.required_batch_size_multiple=1
        total_sequences = lm_model.sample(prompt, beam=1, max_len_a=1, max_len_b=max_tokens, nbest=1, sampling=True, sampling_topp=top_p, temperature=0.6)
        total_sequences = collate_tokens([lm_model.encode(sentence) for sentence in total_sequences], pad_idx=pad_idx)

        fairseq_lm_model = lm_model.models[0]
        with utils.model_eval(fairseq_lm_model):
            logits, extra = fairseq_lm_model(
                total_sequences.to(device=lm_model.device),
                return_all_hiddens=False,
            )

    # -1 for eos token
    # get the top tokens and probs for the context and the generated l tokens
    probs = torch.softmax(logits[:,-max_tokens-1-1:-1], dim=2).cpu()

    top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
    logprobs = torch.log(probs)
    top_log_probs = torch.log(top_probs)

    # construct batched tokens
    # create the return value to resemble OpenAI
    return_json = {}
    for batch_id in range(len(prompt)):
        curr_json = {}
        
        # if you we find one of the stopwords, then we delete everything from the stopword on
        curr_max_tokens = None
        # TODO, implement this for fairseq
        #for stop_word_tensor in stop_words:
        #    if stop_word_tensor[0] in total_sequences[batch_id][-max_tokens:].tolist(): # if stopword is in the list
        #        possible_stop_index = total_sequences[batch_id][-max_tokens:].tolist().index(stop_word_tensor[0]) # get the position of stopword
        #        if possible_stop_index != -1 and total_sequences[batch_id][-max_tokens:].tolist()[possible_stop_index - 1] == 198: # 198 is the \n character. Assert that its in the position before the stopword
        #            if curr_max_tokens is None or possible_stop_index < curr_max_tokens: # save first occruene of stopword
        #                curr_max_tokens = possible_stop_index - 1 # -1 to also cut off \n

        if curr_max_tokens is not None: # stopword is found, cut stuff off
            curr_json['text'] = lm_model.decode(total_sequences[batch_id][-max_tokens-1:-max_tokens + curr_max_tokens])
        else:
            print('no stopword found!') # not having the stopword found is probably a very bad sign
            curr_json['text'] = lm_model.decode(total_sequences[batch_id][-max_tokens-1:-1])

        if curr_max_tokens is None: # no stopword
            curr_top_log_probs = top_log_probs[batch_id][:-1]
            curr_top_tokens = top_tokens[batch_id][:-1]
        else:
            curr_top_log_probs = top_log_probs[batch_id][:curr_max_tokens-1]
            curr_top_tokens = top_tokens[batch_id][:curr_max_tokens-1]
    
        # fill the return json with the top tokens and probs to match the OpenAI return value.
        curr_json['logprobs'] = {}
        curr_json['logprobs']['top_logprobs'] = []
        curr_json['logprobs']['token_logprobs'] = []
        curr_json['logprobs']['tokens'] = []
        for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(curr_top_log_probs, curr_top_tokens)):
            # skip padding tokens
            if current_element_top_tokens[0].item() == pad_idx or current_element_top_tokens[0].item() == eos_idx:
                continue
            temp = {}
            for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                try:
                    temp[lm_model.decode(token.unsqueeze(0))] = log_prob.item()
                except:
                    # madeupwords
                    print('warning made up words')
                    temp[lm_model.string(token.unsqueeze(0))] = log_prob.item()
            curr_json['logprobs']['top_logprobs'].append(temp)

        for index in range(len(probs[batch_id])):
            curr_json['logprobs']['tokens'].append(lm_model.decode(total_sequences[batch_id][index].unsqueeze(0)))
        for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
            # probs are left shifted for LMs
            curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index]])

        choices.append(curr_json)

    return_json['choices'] = choices
    return return_json


def complete(prompt, max_tokens=450, top_p=0.95, **kwargs):
    """complete the prompt using GPT3"""
    openai.api_key = "sk-ujuxY3z8gJPz39dRqVHbnmjUZeIxtlMthwvE6Gm2"
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=max_tokens,
        top_p=top_p,
        #stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
        # can't include print since the API will only handle up to 4 stop words
        stop=["\nclass", "\ndef", "\n#", "\nif"],
        #stop="\ndef",
        logprobs=1,
        **kwargs
    )
    return response

def rank_completions(prompt, num_responses=10, response=None, model_name='davinci-codex'):
    if response is None:
        if 'davinci' in model_name or 'cushman' in model_name:
            response = complete(prompt, n=num_responses)
        if 'fairseq' in model_name:
            response = complete_fairseq_lm(prompt, n=num_responses)
        else:
            response = complete_local_model(prompt, n=num_responses)
    scored_choices = [
        (np.array(choice['logprobs']['token_logprobs']).mean(), choice['text'])
        for choice in response['choices']
    ]
    return list(sorted(scored_choices, reverse=True)), response

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None, required=False)
parser.add_argument("--num_problems", type=int)
parser.add_argument("--num_candidates_generated", type=int, default=15)
parser.add_argument("--num_candidates_evaluated", type=int, default=1)
parser.add_argument("--output_filename", default="samples.jsonl")
parser.add_argument("--response_filename", default="responses.pkl")
parser.add_argument("--cached_responses", action='store_true')
parser.add_argument("--remove_test_cases", default=False, action='store_true')

args = parser.parse_args()

setup_model(args.model_name, args.tokenizer_name)

problems = list(sorted(read_problems().items()))
if args.num_problems is not None:
    problems = problems[:args.num_problems]

samples_to_evaluate = []
if args.cached_responses:
    with open(args.response_filename, 'rb') as f:
        responses = pickle.load(f)
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
    response = responses.get(task_id)
    completions, response = rank_completions(
        prompt, num_responses=args.num_candidates_generated,
        response=response,
        model_name=args.model_name
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

