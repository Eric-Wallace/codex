import os
import time
import numpy as np
import torch
from typing import List

from collections import namedtuple

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
                seq = total_sequences[batch_id][-max_tokens:]
                curr_json = {}

                # if you we find one of the stopwords, then we delete everything from the stopword on
                curr_max_tokens = None
                for stop_word_tensor in encoded_stop_words:
                    for possible_stop_index in len(seq):
                        if torch.allclose(seq[possible_stop_index:len(stop_word_tensor)], stop_word_tensor):
                            if curr_max_tokens is None or possible_stop_index < curr_max_tokens: # save first occurrence of stopword
                                curr_max_tokens = possible_stop_index

                if curr_max_tokens is not None: # stopword is found, cut stuff off
                    curr_json['text'] = self.lm_tokenizer.decode(seq[:curr_max_tokens], skip_special_tokens=True)
                else:
                    print('no stopword found!') # not having the stopword found is probably a very bad sign
                    curr_json['text'] = self.lm_tokenizer.decode(seq, skip_special_tokens=True)


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
    def __init__(self, model_path: str, bpe="gpt2_pretokenization_newlines_only", gpt2_encoder_json=None, gpt2_vocab_bpe=None):
        self.bpe = bpe
        assert bpe in ["gpt2_pretokenization_newlines_only", "gpt2"], f"invalid bpe type {bpe}"
        if not model_path.endswith(".pt"):
            print(f"warning: model_path {model_path} does not end in *.pt")
        model_root_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        if gpt2_encoder_json is None:
            gpt2_encoder_json = os.path.join(model_root_dir, "vocab.json")
        if gpt2_vocab_bpe is None:
            gpt2_vocab_bpe = os.path.join(model_root_dir, "merges.txt")

        from fairseq.models.transformer_lm import TransformerLanguageModel
        self.lm_model = TransformerLanguageModel.from_pretrained(
            model_root_dir, model_basename, bpe=bpe, gpt2_encoder_json=gpt2_encoder_json, gpt2_vocab_bpe=gpt2_vocab_bpe
            )
        self.lm_model.eval().cuda() # TODO do half()

    def encode_stop_words(self, stop_words: List[str]):
        # TODO: I don't think this is needed anymore
        raise NotImplementedError()
        encoded = []
        for stop_word in stop_words:
            # strip the EOS symbol
            enc = self.lm_model.encode(stop_word)
            assert enc[-1] == self.lm_model.src_dict.eos()
            encoded.append(enc[:-1].tolist())
        return encoded

    def complete(self, prompt: str, stop_words: List[str], max_tokens=450, top_p=0.95, n=1, num_log_probs=1, temperature=0.6):
        ''' This function runs fairseq LM locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''
        # need the topk score and token at each step
        # from fairseq.data.data_utils import collate_tokens
        # from fairseq import utils

        if num_log_probs != 1:
            raise NotImplementedError()

        # encoded_stop_words = self.encode_stop_words(stop_words)
        lm_model = self.lm_model

        # pad_idx, eos_idx, bos_idx = lm_model.src_dict.pad(), lm_model.src_dict.eos(), lm_model.src_dict.bos()

        # greedily generate l tokens
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        choices = []
        with torch.no_grad():
            # TODO, for some reason I can't get the sample() to actually return more than one candidaete
            # lm_model.args.batch_size=n
            # lm_model.args.required_batch_size_multiple=1
            # total_sequences = lm_model.sample(
            #     prompt, beam=1, max_len_a=1, max_len_b=max_tokens, nbest=1, sampling=True,
            #      sampling_topp=top_p, temperature=temperature
            #      )
            # total_sequences = collate_tokens([lm_model.encode(sentence) for sentence in total_sequences], pad_idx=pad_idx)

            # fairseq_lm_model = lm_model.models[0]
            # with utils.model_eval(fairseq_lm_model):
            #     logits, extra = fairseq_lm_model(
            #         total_sequences.to(device=lm_model.device),
            #         return_all_hiddens=False,
            #     )
            lm_model.cfg.generation['max_len_b'] = max_tokens
            encoded_prompt = lm_model.encode(prompt)

            prompt_len = len(encoded_prompt)

            # beam is actually just the num of candidates to sample, when sampling=True
            completions = lm_model.generate([encoded_prompt], sampling=True, sampling_topp=top_p, temperature=temperature, beam=n)
            # batch size 1
            completions = completions[0]
            # -1 to remove EOS, both from encoded prompt and from completion
            all_tokens = [completion['tokens'][prompt_len-1:-1] for completion in completions]
            # TODO: these scores are post-temperature-scaling log probs. is this consistent with HF and codex?
            all_logprobs = [completion['positional_scores'][prompt_len-1:-1] for completion in completions]

        # -1 for eos token
        # get the top tokens and probs for the context and the generated l tokens
        # probs = torch.softmax(logits[:,-max_tokens-1-1:-1], dim=2).cpu()

        # top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        # logprobs = torch.log(probs)
        # top_log_probs = torch.log(top_probs)

        # construct batched tokens
        # create the return value to resemble OpenAI
        return_json = {}
        for completion_ix in range(len(completions)):
            curr_json = {}

            # remove EOS
            # seq = total_sequences[batch_id][-max_tokens-1:-1]
            full_seq = all_tokens[completion_ix].cpu()
            full_logprobs = all_logprobs[completion_ix]
            assert len(full_seq) == len(full_logprobs)

            # search for stopwords, to truncate after them
            full_seq_decoded = lm_model.decode(full_seq)
            min_index = None
            for stop_word in stop_words:
                index = full_seq_decoded.find(stop_word)
                if index < 0:
                    continue
                if min_index is None or index < min_index:
                    min_index = index
            
            if index is not None:
                # if you we find one of the stopwords, then we delete everything from the stopword on
                seq_decoded = full_seq_decoded[:min_index]
                # figure out how many tokens to take from log probs by reencoding the truncated string
                # TODO: this may not exactly be right since this I don't think BPE is a prefix code
                # -1 to remove EOS
                seq = lm_model.encode(seq_decoded)[:-1]
                logprobs = full_logprobs[:len(seq)]
            else:
                print('no stopword found!') # not having any stopword found is probably a very bad sign
                seq = full_seq
                seq_decoded = full_seq_decoded

            curr_json['text'] = seq_decoded
            
            # if curr_max_tokens is None: # no stopword
            #     curr_top_log_probs = top_log_probs[batch_id][:-1]
            #     curr_top_tokens = top_tokens[batch_id][:-1]
            # else:
            #     curr_top_log_probs = top_log_probs[batch_id][:curr_max_tokens-1]
            #     curr_top_tokens = top_tokens[batch_id][:curr_max_tokens-1]
        
            # fill the return json with the top tokens and probs to match the OpenAI return value.
            curr_json['logprobs'] = {}
            # curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = logprobs.tolist() 
            curr_json['logprobs']['tokens'] = [lm_model.decode([ix]) for ix in seq]
            # for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(curr_top_log_probs, curr_top_tokens)):
            #     # skip padding tokens
            #     if current_element_top_tokens[0].item() == pad_idx or current_element_top_tokens[0].item() == eos_idx:
            #         continue
            #     temp = {}
            #     for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
            #         try:
            #             temp[lm_model.decode(token.unsqueeze(0))] = log_prob.item()
            #         except:
            #             # madeupwords
            #             print('warning made up words')
            #             temp[lm_model.string(token.unsqueeze(0))] = log_prob.item()
            #     curr_json['logprobs']['top_logprobs'].append(temp)

            # for index in range(len(probs[batch_id])):
            #     curr_json['logprobs']['tokens'].append(lm_model.decode(total_sequences[batch_id][index].unsqueeze(0)))
            # for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
            #     # probs are left shifted for LMs
            #     curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index]])

            choices.append(curr_json)

        return_json['choices'] = choices
        return return_json


class CodeGPT2(Model):
    def __init__(self, model_path='/private/home/fhs/models/pretrained/codemodels/code-gpt2/pytorch_model.bin'):
        from transformers import GPT2LMHeadModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        model.config.max_length = model.config.n_positions 
        model.config.pad_token_id = model.config.eos_token_id
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        # parallellize for multi-GPU training
        n_devices = torch.cuda.device_count()
        layers_per_device = model.config.n_layer // n_devices + 1
        device_map = {k: [i for i in range(layers_per_device * k, min(layers_per_device * (k+1), model.config.n_layer))] for k in range(n_devices)}
        model.parallelize(device_map)
        self.model = model

    def complete(self, prompt, stop_words, n=1, **kwargs):
        # TODO: don't ignore stop words
        if n != 1:
            raise NotImplementedError()
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        if inputs.input_ids.shape[1] > self.model.config.max_length:
            inputs = inputs[:,:self.model.config.max_length]
        generation_output = self.model.generate(**inputs, return_dict_in_generate=True, **kwargs)
        choices = []
        for seq in generation_output['sequences']:
            decoded = self.tokenizer.decode(seq.tolist())
            decoded = decoded.replace('<|endoftext|>', '')
            choices.append({
                'text': decoded
            })
        # TODO: implement logprobs and other stuff or combine this with HFModel
        return {
            'choices': choices
        }

class OpenAIModel(Model):

    def __init__(self, engine='davinci-codex', persistent=True):
        self.engine = engine
        self.persistent = persistent

    def encode_stop_words(self, stop_words: List[str]):
        return stop_words

    def complete(self, prompt, stop_words, max_tokens=450, top_p=0.95, temperature=0.6, **kwargs):
        import openai
        from secret import API_KEY
        openai.api_key = API_KEY

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
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
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
    elif 'fairseq' or '/checkpoint' in model_name:
        return FairseqModel(model_name)
    elif model_name == 'code-gpt2':
        return CodeGPT2()
    else:
        return HFModel(model_name, tokenizer_name)