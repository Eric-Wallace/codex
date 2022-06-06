import os
import sys
import json
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

from typing import List

from tokenizers import ByteLevelBPETokenizer

# root_dir="/checkpoint/dpf/models/cm-6B-armen-none/"
# fname = "checkpoint_best_consolidated.pt"

class InfillingModel:
    TOKENIZER_OFFSET = 4

    EOSS = "<eoss>"

    def __init__(self, model_path, bpe="gpt2_pretokenization_newlines_only"):
        self.model_path = model_path
        root_dir, fname = os.path.split(model_path)
        self.root_dir, self.fname = root_dir, fname
        self.bpe = bpe

        self.tokenizer = tokenizer = ByteLevelBPETokenizer.from_file(
            os.path.join(root_dir, "vocab.json"),
            os.path.join(root_dir, "merges.txt"),
            pretokenizer_split_newlines_only=(bpe=="gpt2_pretokenization_newlines_only")
        )

        print(f"loading model from {root_dir}/{fname}")
        model = TransformerLanguageModel.from_pretrained(root_dir, fname, bpe=bpe, gpt2_encoder_json=f"{root_dir}/vocab.json", gpt2_vocab_bpe=f"{root_dir}/merges.txt")
        model = model.half()
        model = model.cuda().eval()
        self.model = model
        is_cm = True

        if is_cm:
            special_tokens = []
            for i in range(256):
                special_tokens.append(self.make_sentinel(i))
            special_tokens.append(self.EOSS)
            tokenizer.add_special_tokens(special_tokens)

        # set the max generation length
        model.cfg.generation['max_len_b'] = 500

        self.EOSS_ID = tokenizer.token_to_id(self.EOSS) + self.TOKENIZER_OFFSET

    def encode(self, s):
        return torch.tensor(self.tokenizer.encode(s).ids) + self.TOKENIZER_OFFSET

    def decode(self, token_ids):
        token_ids = torch.tensor(token_ids)
        return self.tokenizer.decode((token_ids - self.TOKENIZER_OFFSET).tolist(), skip_special_tokens=False)

    @staticmethod
    def make_sentinel(i):
        return f"<sentinel:{i}>"

    def sentinel_id(self, i):
        return self.tokenizer.token_to_id(self.make_sentinel(i)) + self.TOKENIZER_OFFSET

    def complete(self, s, **kwargs):
        """ complete the prefix s autoregressively """
        model = self.model
        tokenizer = self.tokenizer
        with torch.no_grad():
            #encoded = model.encode(s)
            encoded = torch.tensor(tokenizer.encode(s).ids) + 4
            #encoded = encoded.cuda()
            completion = model.generate([encoded], **kwargs)[0][0]['tokens']
            completion = (completion - 4)[:-1]
            return tokenizer.decode(completion.cpu().tolist(), skip_special_tokens=False)
            #return model.decode(completion)

    def infill(self, parts: List[str], verbose=False, **kwargs):
        # Force the model to fill in code in between each string in parts
        # see code_to_docstring and docstring_to_code for example usages
        model = self.model
        tokenizer = self.tokenizer
        assert isinstance(parts, list)
        infills = []
        if len(parts) == 1:
            return self.complete(parts[0])[len(parts[0]):]

        ids = []

        # encode parts separated by sentinel
        for sentinel_ix, part in enumerate(parts):
            part_tokens = self.encode(part)
            ids.extend(part_tokens.tolist())
            if sentinel_ix < len(parts) - 1:
                ids.append(self.sentinel_id(sentinel_ix))

        infills = []

        complete = []

        # autoregressively fill in
        for sentinel_ix, part in enumerate(parts[:-1]):
            ids.append(self.sentinel_id(sentinel_ix))
            if verbose:
                print(part, end="")
                print(f"<sentinel:{sentinel_ix}>", end="")
            with torch.no_grad():
                completion = model.generate([torch.tensor(ids)], **kwargs)[0][0]['tokens'].tolist()
                if completion[-1] == 2:
                    completion = completion[:-1]

            completion = completion[len(ids):]

            if self.EOSS_ID in completion:
                completion = completion[:completion.index(self.EOSS_ID)+1]
            else:
                if not verbose:
                    print(f"warning: {self.EOSS} not found", file=sys.stderr)
                completion = completion + [self.EOSS_ID]

            ids.extend(completion)

            decoded = self.decode(completion[:-1])
            complete.append(part)
            complete.append(decoded)
            infills.append(decoded)

        complete.append(parts[-1])

        if verbose:
            print(parts[-1])
            print("-"*20)
            print(''.join((complete)))
        return {
            'complete': complete,
            'infills': infills,
            'ids': ids,
            'raw': self.decode(ids)
        }

def code_to_docstring(infilling_model, **kwargs):
    header = '''def count_words(filename):
    "'''

    body = '''"
    counts = Counter()
    with open(filename) as file:
        for line in file:
            words = line.split(' ')
            counts.update(words)
    return counts\n<|/ file |>'''
    return infilling_model.infill([header, body], **kwargs)

def docstring_to_code(infilling_model, **kwargs):
    return infilling_model.infill(['def ', '    """Count the number of occurrences of each word in the file."""\n', '<|/ file |>'], **kwargs)

if __name__ == "__main__":
    infilling_model = InfillingModel("/checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/best.pt")
    #infilling_model = InfillingModel("/checkpoint/dpf/models/cm-1.3B-gpt2tok-xlmg/checkpoint_best_consolidated.pt", bpe="gpt2")
    _ = code_to_docstring(infilling_model, verbose=True, sampling=True, sampling_topp=0.6, temperature=0.6)
