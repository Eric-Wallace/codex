import os
import time
from unicodedata import bidirectional
import numpy as np
from typing import List
import random
import sys
import argparse

from collections import namedtuple
try:
    import torch
except:
    print("couldn't import torch; won't be able to use most models", file=sys.stderr)

from utils import truncate_overlap, truncate_num_lines, stripped_line_split

DEFAULT_MAX_TOKENS = 450

CODEX_RETRY_DELAY_SECONDS = 60
CODEX_MAX_RETRIES = 30

_TruncationParameters = namedtuple("_TruncationParameters", ["max_num_lines", "suffix"])
class TruncationParameters(_TruncationParameters):
    SUFFIX_NUM_CONSECUTIVE_LINES = 2

    HEURISTICS = ["num_lines", "suffix"]

    @staticmethod
    def from_heuristics(truncation_heuristics: List[str], missing_lines: str, suffix: str):
        tp = TruncationParameters(None, None)
        for heuristic in truncation_heuristics:
            assert heuristic in TruncationParameters.HEURISTICS
            if heuristic == "num_lines":
                num_lines = len(stripped_line_split(missing_lines))
                tp = tp._replace(max_num_lines=num_lines)
            elif heuristic == "suffix":
                tp = tp._replace(suffix=suffix)
            else:
                raise NotImplementedError(f"heuristic {heuristic}")
        return tp

    def truncate(self, infill: str):
        """
        Truncate an infill either to the maximum
        """
        infill_truncated = infill
        if self.suffix is not None:
            infill_truncated = truncate_overlap(infill, self.suffix, minimum_num_suffix_lines=self.SUFFIX_NUM_CONSECUTIVE_LINES)
        if self.max_num_lines is not None:
            infill_truncated = truncate_num_lines(infill_truncated, max_num_lines=self.max_num_lines)
        return infill_truncated

def add_model_args(parser):
    parser.add_argument("--model_name", type=str, help="either the name of a codex engine, or a path to a fairseq or HF transformers serialized model. type will be inferred based on the name")
    parser.add_argument("--tokenizer_name", type=str, choices=["gpt2", "gpt2_pretokenization_newlines_only"])
    parser.add_argument("--temperature", type=float, default=0.6, help="pass 0.0 to do greedy or beam decoding")
    parser.add_argument("--top_p", type=float, default=0.95, help="nucleus top-p")
    parser.add_argument("--beam", type=int, default=1, help="beam size; only used if --temperature==0.0")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--prompt_prefix")
    parser.add_argument("--candidate_scoring", choices=["mean", "sum", "random"], default="mean")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)

def add_infilling_args(parser):
    parser.add_argument("--truncation_heuristics", nargs='*', choices=TruncationParameters.HEURISTICS, default=["num_lines"])
    parser.add_argument("--bidirectional_generation", action="store_true", help="for infilling, generate candidates using both left and right contexts")
    parser.add_argument("--bidirectional_scoring", action="store_true", help="for infilling, rerank generated candidates using the left and right contexts")
    parser.add_argument("--num_candidates", type=int, default=10, help="number of candidates to use in infilling reranking")

class Model:
    def encode_stop_words(self, stop_words: List[str]):
        raise NotImplementedError()

    def complete(self, prompt: str, stop_words: List[str], **kwargs):
        text = 'DUMMY'
        choice = {
            'text': text,
            'logprobs': {
                'token_logprobs': None,
                'tokens': None,
            },
        }

        return {
            'prompt': prompt,
            'choices': [choice] * kwargs.get("n", 1),
        }

    def infill(self, parts: List[str], verbose=False, **kwargs):
        # fill in text between each string in parts
        infill = 'DUMMY'
        choice = {
            'complete': [parts[0], infill, parts[1]],
            'infills_untruncated': [infill],
            'ids': None,
            'raw': None,
            'logprobs': {
                'token_logprobs': None,
                'tokens': None,
            },
        }

        return {
            'prompt_parts': parts,
            'choices': [choice] * kwargs.get("n", 1),
        }

    def score_text(self, text_batch: List[str], scoring: str):
        # get the log probability of producing the given text autoregressively
        assert scoring in ['mean', 'sum']
        raise NotImplementedError()

    def _rank_helper(self, choices, scoring):
        assert scoring in ['mean', 'sum', 'random']
        def scoring_fn(choice):
            if scoring == 'random':
                return random.random()
            token_logprobs = choice['logprobs']['token_logprobs']
            token_logprobs = np.array(token_logprobs)
            if scoring =='mean':
                return token_logprobs.mean()
            elif scoring == 'sum':
                return token_logprobs.sum()
            else:
                raise NotImplementedError(f"scoring {scoring}")
        return list(sorted(choices, key=scoring_fn, reverse=True))

    def rank_completions(self, prompt: str, stop_words: List[str], cached_response=None, scoring='mean', sampling=True, temperature=0.6, top_p=0.95, n=1, max_tokens=DEFAULT_MAX_TOKENS, beam=1):
        if cached_response is None:
            response = self.complete(prompt, stop_words, sampling=sampling, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens, beam=beam)
        else:
            response = cached_response
        sorted_choices = self._rank_helper(response['choices'], scoring=scoring)
        return sorted_choices, response

    def rank_infills(self, parts: List[str], verbose=False, bidirectional_scoring=False, bidirectional_generation=False,
                    cached_response=None, scoring='mean',
                    truncation_parameters: List[TruncationParameters] = None,
                    sampling=True, temperature=0.6, top_p=0.95, n=1, max_tokens=DEFAULT_MAX_TOKENS, beam=1):
        if truncation_parameters is None:
            truncation_parameters = [TruncationParameters(None, None) for _ in parts[:-1]]
        assert len(truncation_parameters) == len(parts) - 1

        if len(parts) != 2:
            # TODO: implement this
            raise NotImplementedError()
        else:
            prefix = parts[0]
            suffix = parts[1]
            trunc_params = truncation_parameters[0]

        if cached_response is None:
            if bidirectional_generation:
                response = self.infill([prefix, suffix], truncation_parameters=[trunc_params], verbose=verbose, sampling=sampling, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens, beam=beam)
            else:
                response = self.complete(prefix, stop_words=[], sampling=sampling, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens, beam=beam)
        else:
            response = cached_response

        choices = []
        for choice in response['choices']:
            if bidirectional_generation:
                infills_untruncated = choice['infills_untruncated']
            else:
                infills_untruncated = [choice['text']]
            assert len(infills_untruncated) == 1
            text_untruncated = infills_untruncated[0]
            text = trunc_params.truncate(text_untruncated)
            if verbose:
                print(f"--prefix:--\n{prefix}")
                print(f"--infill (truncated):--\n{text}")
                print(f"--infill (untruncated):--\n{text_untruncated}")
                print(f"--suffix:--\n{suffix}")

            def maybe_append_newline(s):
                if not s.endswith("\n"):
                    return s + "\n"
                return s
            
            d = {
                'complete': ''.join([maybe_append_newline(prefix), maybe_append_newline(text), suffix]),
                'infills': [text],
                'infills_untruncated': infills_untruncated,
                'logprobs': {
                    'token_logprobs': None,
                    'tokens': None,
                }
            }
            choices.append(d)

        if bidirectional_scoring:
            completes = [choice['complete'] for choice in choices]
            scores = self.score_text(completes, scoring=scoring)
            assert len(scores) == len(choices)
            for choice, score in zip(choices, scores):
                choice['complete_score'] = score
                choice['complete_score_method'] = scoring
            sorted_choices = list(sorted(choices, key=lambda d: d['complete_score'], reverse=True))
        else:
            sorted_choices = self._rank_helper(choices, scoring)

        return sorted_choices, response

class HFModel(Model):
    def __init__(self, model_name, tokenizer_name=None, prompt_prefix=None, batch_size=None):
        if prompt_prefix is not None:
            raise NotImplementedError("--prompt_prefix for HFModel")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        #self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        from transformers import GPTJForCausalLM
        self.lm_model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.lm_model.eval().cuda()
    
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.lm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.batch_size = batch_size

    def encode_stop_words(self, stop_words: List[str]):
        return [self.lm_tokenizer.encode(string) for string in stop_words]

    def complete(self, prompt, stop_words: List[str], sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, n=1, num_log_probs=1, temperature=0.6, beam=1):
        ''' This function runs GPT-2 locally using HF transformers but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''

        batch_size = n if self.batch_size is None else self.batch_size

        if beam != 1:
            raise NotImplementedError()

        if not sampling:
            raise NotImplementedError()

        assert isinstance(prompt, str)
        prompt = [prompt] # below code assumes list

        encoded_stop_words = self.encode_stop_words(stop_words)
        # print(f"stop_words: {stop_words}")
        # print(f"encoded_stop_words: {encoded_stop_words}")

        input_ids = self.lm_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=False)
        choices = []
        lm_model = self.lm_model
        while len(choices) < n:
            num_to_sample = min(batch_size, n - len(choices))
            print(f"num_to_sample: {num_to_sample}")
            with torch.inference_mode():
                # generate from the model
                total_sequences = lm_model.generate(
                    input_ids=input_ids['input_ids'].cuda(),
                    attention_mask=input_ids['attention_mask'].cuda(),
                    max_length=max_tokens + len(input_ids['input_ids'][0]),
                    do_sample=True,
                    num_return_sequences=num_to_sample,
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
                    for possible_stop_index in range(len(seq)):
                        if seq[possible_stop_index:possible_stop_index+len(stop_word_tensor)].tolist() == stop_word_tensor:
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
    def __init__(self, args: argparse.Namespace, model_path: str, tokenizer_name=None, gpt2_encoder_json=None, gpt2_vocab_bpe=None, prompt_prefix=None, batch_size=None, model=None):
        self.args = args
        if tokenizer_name is None:
            bpe = "gpt2_pretokenization_newlines_only"
        else:
            bpe = tokenizer_name
        self.bpe = bpe
        assert bpe in ["gpt2_pretokenization_newlines_only", "gpt2"], f"invalid bpe type {bpe}"
        if not model_path.endswith(".pt"):
            print(f"warning: model_path {model_path} does not end in *.pt")
        assert os.path.exists(model_path), f"model_path {model_path} should be a file"
        model_root_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        if gpt2_encoder_json is None:
            gpt2_encoder_json = os.path.join(model_root_dir, "vocab.json")
        if gpt2_vocab_bpe is None:
            gpt2_vocab_bpe = os.path.join(model_root_dir, "merges.txt")

        self.gpt2_encoder_json = gpt2_encoder_json
        self.gpt2_vocab_bpe = gpt2_vocab_bpe

        print(f"model_root_dir: {model_root_dir}")
        print(f"model_basename: {model_basename}")
        from fairseq.models.transformer_lm import TransformerLanguageModel
        if model is None:
            self.lm_model = TransformerLanguageModel.from_pretrained(
                model_root_dir, model_basename, bpe=bpe, gpt2_encoder_json=gpt2_encoder_json, gpt2_vocab_bpe=gpt2_vocab_bpe
                ).half()
            self.lm_model.eval().cuda() 
        else:
            self.lm_model = model

        # length normalization? 
        #self.unnormalized = args.unnormalized
        self.unnormalized = True
        self.lm_model.cfg.generation['unnormalized'] = True

        self.prompt_prefix = prompt_prefix
        self.eos_index = self.lm_model.task.dictionary.eos_index

        self.batch_size = batch_size

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

    @property
    def _extra_stop_words(self):
        return ["<| ", "<|/ ", "<code>", "</code>", "<cell>", "</cell>"]

    def _encode(self, text: str):
        # -> torch.tensor
        return self.lm_model.encode(text)

    def _decode(self, tokens) -> str:
        # tokens: torch.tensor
        return self.lm_model.decode(tokens)

    def score_text(self, text_batch: List[str], scoring: str='sum'):
        tokens = [self._encode(text) for text in text_batch]
        return self.score_tokens(tokens, scoring)

    def score_tokens(self, tokens_batch, scoring='sum'):
        # tokens_batch: List[torch.tensor]
        # not sure if passing temperature here does anything, but it shouldn't hurt
        all_scores = []

        i = 0
        while len(all_scores) < len(tokens_batch):
            subbatch = tokens_batch[i:i+self.batch_size]
            i += self.batch_size
            ret_vals = self.lm_model.generate(subbatch, score_reference=True, temperature=1.0)
            for ret_val in ret_vals:
                assert len(ret_val) == 1
                log_probs = ret_val[0]['positional_scores']
                if scoring == 'sum':
                    score = log_probs.sum()
                elif scoring == 'mean':
                    score = log_probs.mean()
                else:
                    raise NotImplementedError(f"scoring {scoring}")
                all_scores.append(score.item())
        return all_scores

    def _generate(self, encoded_prompt: torch.tensor, max_tokens, top_p=0.95, n=1, temperature=0.6):
        assert encoded_prompt.dim() == 1
        prompt_len = len(encoded_prompt)

        from fairseq.priming_generator import GreedyDecoding, TopPSampling
        # strip EOS from end
        if encoded_prompt[-1].item() == self.eos:
            encoded_prompt = encoded_prompt[:-1]
        if temperature == 0:
            decoder = GreedyDecoding(self.lm_model, min_len=prompt_len, max_len=max_tokens+prompt_len, temperature=temperature, show_tqdm=False)
        else:
            decoder = TopPSampling(self.lm_model, min_len=prompt_len, max_len=max_tokens+prompt_len, sampling_topp=top_p, temperature=temperature, show_tqdm=False)

        decoder = decoder.to(self.lm_model.device)
        encoded_prompt = encoded_prompt.to(self.lm_model.device)

        all_tokens = []
        all_log_probs = []
        while len(all_tokens) < n:
            completion_tokens, completion_token_log_probs = decoder.decode(
                encoded_prompt.to(self.lm_model.device),
                return_log_probs=True,
                stop_on_eos=True
            )
            # remove initial EOS
            assert completion_tokens[0].item() == self.eos
            completion_tokens = completion_tokens[1:]
            completion_token_log_probs = completion_token_log_probs[1:]
            assert torch.allclose(completion_tokens[:prompt_len], encoded_prompt)
            if completion_tokens[-1] == self.eos:
                completion_tokens = completion_tokens[:-1]
                completion_token_log_probs = completion_token_log_probs[:-1]
            assert completion_tokens.size() == completion_token_log_probs.size()
            all_tokens.append(completion_tokens[prompt_len:])
            all_log_probs.append(completion_token_log_probs)
        return all_tokens, all_log_probs

    def complete(self, prompt: str, stop_words: List[str], sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, n=1, num_log_probs=1, temperature=0.6, beam=1):
        ''' This function runs fairseq LM locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''

        assert beam == 1, "beam search is not implemented"

        batch_size = n if self.batch_size is None else self.batch_size

        if not sampling:
            raise NotImplementedError()

        if num_log_probs != 1:
            raise NotImplementedError()

        stop_words = stop_words + self._extra_stop_words

        if self.prompt_prefix is not None:
            # TODO: add option to not insert newline
            prompt = f"{self.prompt_prefix}\n{prompt}"

        encoded_prompt = self._encode(prompt)

        all_tokens, all_log_probs = self._generate(
            encoded_prompt: torch.tensor, 
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            temperature=temperature,
        )

        assert len(all_tokens) == n
        assert len(all_log_probs) == n

        # construct batched tokens
        # create the return value to resemble OpenAI
        choices = []
        return_json = {}
        for completion_ix in range(len(all_tokens)):
            curr_json = {}

            # remove EOS
            # seq = total_sequences[batch_id][-max_tokens-1:-1]
            full_seq = all_tokens[completion_ix].cpu()
            full_logprobs = all_logprobs[completion_ix]
            assert len(full_seq) == len(full_logprobs)

            # search for stopwords, to truncate after them
            full_seq_decoded = self._decode(full_seq)
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
                seq = self._encode(seq_decoded)[:-1]
                logprobs = full_logprobs[:len(seq)]
            else:
                print('no stopword found!') # not having any stopword found is probably a very bad sign
                seq = full_seq
                seq_decoded = full_seq_decoded

            curr_json['text'] = seq_decoded
            
        
            # fill the return json with the top tokens and probs to match the OpenAI return value.
            curr_json['logprobs'] = {}
            curr_json['logprobs']['token_logprobs'] = logprobs.tolist() 
            curr_json['logprobs']['tokens'] = [self._decode([ix]) for ix in seq]

            # TODO: add top_logprobs
            # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}

            choices.append(curr_json)

        # {'choices': [{'text': text, 'logprobs': ... }]}
        return_json['choices'] = choices
        return return_json

class CausalMasking(FairseqModel):
    EOSS = "<eoss>"

    TOKENIZER_OFFSET = 4

    @staticmethod
    def make_sentinel(i):
        return f"<sentinel:{i}>"

    def sentinel_id(self, i):
        return self.tokenizer.token_to_id(self.make_sentinel(i)) + self.TOKENIZER_OFFSET

    @property
    def _sentinel_tokens(self):
        return [CausalMasking.make_sentinel(i) for i in range(256)]

    @property
    def _special_tokens(self):
        return self._sentinel_tokens + [self.EOSS]

    @property
    def _extra_stop_words(self):
        return super()._extra_stop_words + self._special_tokens

    def __init__(self, args, model_path: str, bpe="gpt2_pretokenization_newlines_only", gpt2_encoder_json=None, gpt2_vocab_bpe=None, prompt_prefix=None, batch_size=None, model=None):
        super().__init__(args, model_path, bpe, gpt2_encoder_json, gpt2_vocab_bpe, prompt_prefix, batch_size=batch_size, model=model)
        assert bpe in {"gpt2_pretokenization_newlines_only", "bpe"}
        from tokenizers import ByteLevelBPETokenizer
        self.tokenizer = tokenizer = ByteLevelBPETokenizer.from_file(
            # these will be set by super().__init__
            self.gpt2_encoder_json, self.gpt2_vocab_bpe,
            pretokenizer_split_newlines_only=(bpe=="gpt2_pretokenization_newlines_only"),
        )
        tokenizer.add_special_tokens(self._special_tokens)

        self.EOSS_ID = tokenizer.token_to_id(self.EOSS) + self.TOKENIZER_OFFSET

    def _encode(self, text, include_eos=True):
        if include_eos:
            return torch.tensor(self.tokenizer.encode(text).ids + [self.eos_index - self.TOKENIZER_OFFSET]) + self.TOKENIZER_OFFSET
        else:
            return torch.tensor(self.tokenizer.encode(text).ids) + self.TOKENIZER_OFFSET

    def _decode(self, tokens):
        token_ids = torch.tensor(tokens)
        return self.tokenizer.decode((token_ids - self.TOKENIZER_OFFSET).tolist(), skip_special_tokens=False)

    def infill(self, parts: List[str], verbose=False, n=1, truncation_parameters: List[TruncationParameters]=None, sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, temperature=0.0, beam=1):
        # Force the model to fill in code in between each string in parts
        # see code_to_docstring and docstring_to_code for example usages
        if truncation_parameters is None:
            truncation_parameters = [TruncationParameters(None, None) for _ in parts[:-1]]

        assert len(truncation_parameters) == len(parts) - 1
        model = self.lm_model
        assert isinstance(parts, list)
        assert len(parts) > 1

        if self.prompt_prefix is not None:
            raise NotImplementedError()
            parts = parts.copy()
            parts[0] = f"{self.prompt_prefix}\n{parts[0]}"

        infills = []

        ids = []
        
        if n != 1:
            raise NotImplementedError()

        # encode parts separated by sentinel
        for sentinel_ix, part in enumerate(parts):
            part_tokens = self._encode(part, include_eos=False)
            ids.extend(part_tokens.tolist())
            if sentinel_ix < len(parts) - 1:
                ids.append(self.sentinel_id(sentinel_ix))

        infills = []

        complete = []

        infill_scores = []
        infill_tokens = []

        # autoregressively fill in
        for sentinel_ix, part in enumerate(parts[:-1]):
            ids.append(self.sentinel_id(sentinel_ix))
            model.cfg.generation['max_len_b'] = max_tokens + len(ids)
            if verbose:
                print(part, end="")
                print(f"<sentinel:{sentinel_ix}>", end="")
            with torch.no_grad():
                # print("completing: ")
                # print(self._decode(ids))
                if temperature == 0:
                    assert n==1
                    # print("not sampling")
                    outputs = model.generate(
                        [torch.tensor(ids)], sampling=False, beam=beam,
                    )
                else:
                    # TODO: batch
                    assert beam == 1, "cannot have a non-zero temperature and beam != 1"
                    # the beam argument to generate actually just specifies the num of candidates to sample, when sampling=True
                    outputs = model.generate(
                        [torch.tensor(ids)], sampling=True, beam=n, sampling_topp=top_p, temperature=temperature,
                    )
                completion = outputs[0][0]['tokens'].tolist()
                scores = outputs[0][0]['positional_scores']
                if completion[-1] == 2:
                    completion = completion[:-1]
                    scores = scores[:-1]

            # TODO: replace the code above with a call to _generate, making sure to check that EOS are accounted for in the prefix removal below (they may have been added, so len(ids) might not be the right thing to use? maybe other stuff too)

            completion = completion[len(ids):]
            scores = scores[len(ids):]

            if self.EOSS_ID in completion:
                t = completion.index(self.EOSS_ID)+1
                completion = completion[:t]
                scores = scores[:t]
                # TODO: handle this better: we do want to include the score for EOSS (if we use these scores somewhere)
                # but how do we handle the case where EOSS is not present (below) without biasing toward those candidates?
                scores = scores[:-1]
            else:
                if not verbose:
                    print(f"warning: {self.EOSS} not found", file=sys.stderr)
                completion = completion + [self.EOSS_ID]

            ids.extend(completion)

            infill_tokens.append(completion[:-1])
            infill_scores.append(scores)

            decoded = self._decode(completion[:-1])
            # print("decoded:")
            # print(decoded)
            # print()
            complete.append(part)
            complete.append(decoded)
            infills.append(decoded)

        complete.append(parts[-1])

        # if verbose:
        #     print(parts[-1])
        #     print("-"*20)
        #     print(''.join((complete)))

        # decoded_tokens = [ [self._decode([t]) for t in completion] for completion in infill_tokens]
        # print("complete:")
        # print(complete)
        # print()

        choice = {
            'complete': complete,
            'infills_untruncated': infills,
            'ids': ids,
            # 'raw': self._decode(ids),
            'logprobs': {
                'token_logprobs': infill_scores,
                'tokens': None,
            },
        }

        return {
            'prompt_parts': parts,
            'choices': [choice],
        }


class CodeGPT2(Model):
    def __init__(self, model_path='/private/home/fhs/models/pretrained/codemodels/code-gpt2/pytorch_model.bin', prompt_prefix=None):
        raise NotImplementedError("TODO: update complete to match new method sigs")
        if prompt_prefix is not None:
            raise NotImplementedError("--prompt_prefix for CodeGPT2")
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

    def __init__(self, engine='davinci-codex', persistent=True, prompt_prefix=None):
        if prompt_prefix is not None:
            raise NotImplementedError("--prompt_prefix for OpenAIModel")
        self.engine = engine
        self.persistent = persistent

    def encode_stop_words(self, stop_words: List[str]):
        return stop_words

    def score_text(self, text_batch: List[str], scoring: str):
        all_scores = []
        for text in text_batch:
            response = self.complete(text, None, max_tokens=0, temperature=1.0, echo=True)
            choice = response['choices'][0]
            log_probs = choice["logprobs"]['token_logprobs'][1:]
            if scoring == 'sum':
                score = np.sum(log_probs)
            elif scoring == 'mean':
                score = np.mean(log_probs)
            else:
                raise NotImplementedError(f"scoring {scoring}")
            all_scores.append(score)
        return all_scores

    def complete(self, prompt, stop_words, max_tokens=450, top_p=0.95, temperature=0.6, sampling=True, beam=1, **kwargs):
        if stop_words == []:
            stop_words = None
        if not sampling:
            raise NotImplementedError()
        if beam != 1:
            raise NotImplementedError()
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
                    engine=self.engine,
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

def make_model(args, cached_model=None):
    model_name = args.model_name
    if model_name is None:
        return Model()
    tokenizer_name = args.tokenizer_name
    prompt_prefix = args.prompt_prefix
    if 'davinci' in model_name or 'cushman' in model_name:
        if prompt_prefix is not None:
            raise NotImplementedError("prompt prefix for codex models")
        return OpenAIModel(model_name, persistent=True)
    elif 'fairseq' or '/checkpoint' in model_name:
        if 'cm-' in model_name:
            return CausalMasking(args, model_name, prompt_prefix=prompt_prefix, batch_size=args.batch_size, model=cached_model)
        elif 'lm-' in model_name:
            return FairseqModel(args, model_name, prompt_prefix=prompt_prefix, batch_size=args.batch_size, model=cached_model)
        elif 'gpt-j' in model_name:
            if prompt_prefix is not None:
                raise NotImplementedError()
            return HFModel(model_name, tokenizer_name, batch_size=args.batch_size)
        else:
            raise ValueError(f"couldn't guess model type from {model_name}")
    elif model_name == 'code-gpt2':
        if prompt_prefix is not None:
            raise NotImplementedError()
        return CodeGPT2()
    else:
        if prompt_prefix is not None:
            raise NotImplementedError()
        return HFModel(model_name, tokenizer_name, batch_size=args.batch_size)
