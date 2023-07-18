# Implements simple variants (specifically 1 step look ahead, no batching, no topk) of decoding with a strong focus on prefix correctness.
# Fairseq SequenceGenerator code is to complex to reason about
# author: Armen Aghajanyan

import torch
from torch import nn
from fairseq.data import Dictionary
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer_lm import TransformerLanguageModel
from typing import Dict, Optional, List
from tqdm import tqdm

def forward_encoder(model, net_input):
    if not hasattr(model, "encoder"):
        return None
    return model.encoder.forward_torchscript(net_input)


def unpack_decoder_out(model, decoder_out, temperature: float):
    attn: Optional[torch.Tensor] = None
    decoder_len = len(decoder_out)
    if decoder_len > 1 and decoder_out[1] is not None:
        if isinstance(decoder_out[1], torch.Tensor):
            attn = decoder_out[1]
        else:
            attn_holder = decoder_out[1]["attn"]
            if isinstance(attn_holder, torch.Tensor):
                attn = attn_holder
            elif attn_holder is not None:
                attn = attn_holder[0]
        if attn is not None:
            attn = attn[:, -1, :]

    decoder_out_tuple = (
        decoder_out[0][:, -1:, :].div_(temperature),
        None if decoder_len <= 1 else decoder_out[1],
    )
    probs = model.get_normalized_probs(
        decoder_out_tuple, log_probs=True, sample=None
    )
    probs = probs[:, -1, :]
    return probs, attn


class DecodingBase(nn.Module):
    def __init__(self, transformer_language_model: GeneratorHubInterface, min_len: int, max_len: int, temperature: float = 1.0, show_tqdm=True):
        super().__init__()
        self.language_model = transformer_language_model
        self.dictionary: Dictionary = transformer_language_model.task.source_dictionary
        self.model = transformer_language_model.models[0]
        self.bos = self.dictionary.bos()
        self.eos = self.dictionary.eos()
        self.pad = self.dictionary.pad()
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.show_tqdm = show_tqdm

    def decode_multiple_candidates(self, prefix: torch.Tensor, num_candidates: int = 1, encoded_stop_words: Optional[List[List[int]]] = None, all_must_complete=True, incremental_states=None, return_incremental_states=False):
        """
        returns: (tokens, token_logprobs)
        tokens: (batch_size x max_seq_length) LongTensor
        token_logprobs: (batch_size x max_seq_length) FloatTensor
        """
        if encoded_stop_words is None:
            encoded_stop_words = []
        encoded_stop_words = [[self.eos]] + encoded_stop_words
        for esw in encoded_stop_words:
            assert isinstance(esw, list)
            assert isinstance(esw[0], int)
        with torch.no_grad():
            prefix = prefix.to(self.dummy_param.device)
            assert prefix.ndim == 1 and prefix.size(0) > 0
            if prefix[0] != self.eos:
                prefix = torch.cat(
                    [torch.tensor([self.eos]).to(prefix), prefix])
            prefix_len: int = prefix.size(0)
            assert prefix_len < self.max_len, "Max len is smaller than prefix length"

            src_tokens = prefix
            print("src_tokens:")
            print(src_tokens)
            print()
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum()
            )
            encoder_out = forward_encoder(self.model, {"src_tokens": src_tokens.unsqueeze(
                0), "src_lengths": src_lengths.unsqueeze(0)})

            if encoder_out is not None:
                encoder_out = encoder_out.view(num_candidates, -1, -1)

            if incremental_states is None:
                incremental_states = torch.jit.annotate(
                    Dict[str, Dict[str, Optional[torch.Tensor]]],
                    torch.jit.annotate(
                        Dict[str, Dict[str, Optional[torch.Tensor]]], {})
                )
                start_step = 1
            else:
                # incremental states are bsz x d1 x prev_timesteps x d2
                cached_size = next(iter(incremental_states.values()))['prev_key'].size()
                assert len(cached_size) == 4
                # we can start decoding at the next timestep
                start_step = cached_size[2] + 1
            tokens = (
                torch.zeros(num_candidates, self.max_len)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
            )
            tokens[:, :len(prefix)] = prefix

            token_log_probs = (
                torch.zeros(num_candidates, self.max_len)
                .to(src_tokens)
                .float()
            )

            seq_lengths = torch.ones(num_candidates).to(src_tokens.device).long()

            it = range(start_step, self.max_len)
            if self.show_tqdm:
                it = tqdm(it)

            found_stop = torch.zeros(num_candidates).to(src_tokens.device).bool()

            for step in it:
                decoder_out = self.model.decoder.forward(
                    tokens[:, :step],
                    encoder_out=encoder_out,
                    incremental_state=incremental_states,
                )
                logprobs, _ = unpack_decoder_out(
                    self.model, decoder_out, self.temperature)
                seq_lengths += (~found_stop).long()
                if step < len(prefix):
                    tokens[:, step] = prefix[step]
                else:
                    for candidate_ix in range(num_candidates):
                        candidate_lps = logprobs[candidate_ix]
                        token = self.choice(candidate_lps)
                        tokens[candidate_ix, step] = token
                        token_log_probs[candidate_ix, step] = candidate_lps[token]
                        candidate_found_stop = False
                        for esw in encoded_stop_words:
                            if tokens[candidate_ix, step+1-len(esw):step+1].tolist() == esw:
                                if step < len(prefix):
                                    print(f"warning: stopping on {token.item()} at step {step} within prefix {prefix}")
                                candidate_found_stop = True
                        found_stop[candidate_ix] |= candidate_found_stop
                if (all_must_complete and found_stop.all()) or ((not all_must_complete) and found_stop.any()):
                    tokens = tokens[:, :step+1]
                    token_log_probs = token_log_probs[:, :step+1]
                    break
            if return_incremental_states:
                return tokens, token_log_probs, seq_lengths, found_stop, incremental_states
            else:
                return tokens, token_log_probs, seq_lengths, found_stop


    def decode(self, prefix: Optional[torch.Tensor] = None, return_log_probs=False, encoded_stop_words: List[List[int]]=None) -> torch.Tensor:
        if prefix is None:
            prefix = torch.tensor([self.eos]).to(self.dummy_param.device)
        tokens, token_log_probs, seq_lengths = self.decode_multiple_candidates(prefix, num_candidates=1, encoded_stop_words=encoded_stop_words)
        if return_log_probs:
            return tokens.squeeze(0), token_log_probs.squeeze(0)
        else:
            return tokens.squeeze(0)

    def choice(self, logprob: torch.Tensor) -> int:
        raise NotImplementedError


class GreedyDecoding(DecodingBase):
    def choice(self, logprob: torch.Tensor) -> int:
        return torch.argmax(logprob)


class TopPSampling(DecodingBase):
    def __init__(self, transformer_language_model: GeneratorHubInterface, min_len: int, max_len: int, sampling_topp: float, temperature: float = 1.0, show_tqdm=True):
        super().__init__(transformer_language_model, min_len, max_len, temperature, show_tqdm)
        self.sampling_topp = sampling_topp

    def choice(self, logprob: torch.Tensor) -> int:
        probs = logprob.exp_()
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=0)
        mask = cumsum_probs.lt(self.sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=0)
        last_included = cumsum_mask[-1:]
        last_included.clamp_(0, mask.size(0) - 1)
        mask = mask.scatter_(0, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[: max_dim + 1]
        truncated_probs = sorted_probs[: max_dim + 1]
        truncated_indices = sorted_indices[: max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)

        indices_buf = torch.multinomial(
            trimed_probs, num_samples=1
        )
        return truncated_indices[indices_buf]


class TopKSampling(DecodingBase):
    def __init__(self, transformer_language_model: GeneratorHubInterface, min_len: int, max_len: int, sampling_topk: int, temperature: float = 1.0):
        super().__init__(transformer_language_model, min_len, max_len, temperature)
        self.sampling_topk = sampling_topk

    def choice(self, logprob: torch.Tensor) -> int:
        lprobs, top_indices = logprob.topk(self.sampling_topk)
        probs = lprobs.exp_()
        indices_buf = torch.multinomial(
            probs,
            1,
            replacement=True,
        )

        return top_indices[indices_buf]
