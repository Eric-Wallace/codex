import time
import numpy as np
import torch
from typing import List

from collections import namedtuple

import openai
from secret import API_KEY
openai.api_key = API_KEY

CODEX_RETRY_DELAY_SECONDS = 60
CODEX_MAX_RETRIES = 30

class Model:
    def encode_stop_words(self, stop_words: List[str]):
        raise NotImplementedError()

    def complete(self, prompt: str, stop_words: List[str], **kwargs):
        raise NotImplementedError()

    def rank_completions(self, prompt: str, stop_words: List[str], **kwargs):
        response = self.complete(prompt, stop_words, **kwargs)
        scored_choices = [
            (np.array(choice['logprobs']['token_logprobs']).mean(), choice['text'])
            for choice in response['choices']
        ]
        return list(sorted(scored_choices, reverse=True)), response

class HFModel(Model):
    def __init__(self, model_name, tokenizer_name=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.lm_model.eval().cuda() # TODO do half()
    
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.lm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode_stop_words(self, stop_words: List[str]):
        return [self.lm_tokenizer.encode(token) for token in stop_words]

    def complete(self, prompt, stop_words: List[str], max_tokens=450, top_p=0.95, n=1, num_log_probs=1, temperature=0.6):
        ''' This function runs GPT-2 locally using HF transformers but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''
        assert isinstance(prompt, str)
        prompt = [prompt] # below code assumes list

        encoded_stop_words = self.encode_stop_words(stop_words)

        input_ids = self.lm_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=False)
        choices = []
        lm_model = self.lm_model
        while len(choices) < n:
            num_to_sample = min(5, n - len(choices))
            with torch.inference_mode():
                # generate from the model
                total_sequences = lm_model.generate(
                    input_ids=input_ids['input_ids'].cuda(),
                    attention_mask=input_ids['attention_mask'].cuda(),
                    max_length=max_tokens + len(input_ids['input_ids'][0]),
                    do_sample=True,
                    num_return_sequences=n,
                    top_p=top_p,
                    early_stopping=True,
                    use_cache=True,
                    temperature=temperature,
                )

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
                for stop_word_tensor in encoded_stop_words:
                    if stop_word_tensor[0] in total_sequences[batch_id][-max_tokens:].tolist(): # if stopword is in the list
                        possible_stop_index = total_sequences[batch_id][-max_tokens:].tolist().index(stop_word_tensor[0]) # get the position of stopword
                        if possible_stop_index != -1 and total_sequences[batch_id][-max_tokens:].tolist()[possible_stop_index - 1] == 198: # 198 is the \n character. Assert that its in the position before the stopword
                            if curr_max_tokens is None or possible_stop_index < curr_max_tokens: # save first occruene of stopword
                                curr_max_tokens = possible_stop_index - 1 # -1 to also cut off \n

                if curr_max_tokens is not None: # stopword is found, cut stuff off
                    curr_json['text'] = self.lm_tokenizer.decode(total_sequences[batch_id][-max_tokens:-max_tokens + curr_max_tokens], skip_special_tokens=True)
                else:
                    print('no stopword found!') # not having the stopword found is probably a very bad sign
                    curr_json['text'] = self.lm_tokenizer.decode(total_sequences[batch_id][-max_tokens:], skip_special_tokens=True)


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
                    curr_json['logprobs']['tokens'].append(self.lm_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[self.lm_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)

                choices.append(curr_json)
        return_json['choices'] = choices
        return return_json

class FairseqModel(Model):
    def __init__(self, model_name):
        from fairseq.hub_utils import from_pretrained, GeneratorHubInterface
        #fairseq_pretrain = from_pretrained('/'.join(model_name.split('/')[0:-1]), 'checkpoint.pt', bpe='gpt2')
        fairseq_pretrain = from_pretrained(model_name, 'model.pt')
        self.lm_model = GeneratorHubInterface(fairseq_pretrain['args'], models=fairseq_pretrain['models'], task=fairseq_pretrain['task'] )
        self.lm_model.eval().cuda() # TODO do half()

    def encode_stop_words(self, stop_words: List[str]):
        return [self.lm_model.encode(token)[0:1].tolist() for token in stop_words]

    def complete(self, prompt: str, stop_words: List[str], max_tokens=450, top_p=0.95, n=1, num_log_probs=1):
        ''' This function runs fairseq LM locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''
        # need the topk score and token at each step
        from fairseq.data.data_utils import collate_tokens
        from fairseq import utils

        encoded_stop_words = self.encode_stop_words(stop_words)
        lm_model = self.lm_model

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


class OpenAIModel(Model):
    def __init__(self, engine='davinci-codex', persistent=True):
        self.engine = engine
        self.persistent = persistent

    def encode_stop_words(self, stop_words: List[str]):
        return stop_words

    def complete(self, prompt, stop_words, **kwargs):
        succeeded = False
        tries = 0
        while not succeeded:
            tries += 1
            if tries > CODEX_MAX_RETRIES:
                raise Exception("max number of retries failed")
            try:
                response = openai.Completion.create(
                    engine="davinci-codex",
                    prompt=prompt,
                    stop=stop_words,
                    logprobs=1,
                    **kwargs
                )
                succeeded = True
            except openai.error.RateLimitError as e:
                if not self.persistent:
                    raise e
                print(e)
                time.sleep(CODEX_RETRY_DELAY_SECONDS)
        return response

def make_model(model_name, tokenizer_name):
    if 'davinci' in model_name or 'cushman' in model_name:
        return OpenAIModel(model_name, persistent=True)
    elif 'fairseq' in model_name:
        return FairseqModel(model_name)
    else:
        return HFModel(model_name, tokenizer_name)