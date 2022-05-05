from pathlib import Path
import json
import shutil

TOKEN_OFFSET = 4
bos_token = "<s>"
pad_token = "<pad>"
eos_token = "<|endoftext|>"
unk_token = "<unk>"

data_dir = Path("tokenizer")
tok_v1_dir = data_dir / "v1"
tok_v2_dir = data_dir / "v2"

with open(tok_v1_dir / "vocab.json", "r") as fi:
    vocab_json = json.load(fi)

for token, id in vocab_json.items():
    vocab_json[token] = id + TOKEN_OFFSET

vocab_json[bos_token] = 0
vocab_json[pad_token] = 1
vocab_json[eos_token] = 2
vocab_json[unk_token] = 3


with open(tok_v2_dir / "vocab.json", "w") as fi:
    json.dump(vocab_json, fi)

src_merges_path = tok_v1_dir / "merges.txt"
tgt_merges_path = tok_v2_dir / "merges.txt"

shutil.copyfile(src_merges_path, tgt_merges_path)
