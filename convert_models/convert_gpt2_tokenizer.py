from pathlib import Path
import json
import shutil

TOKEN_OFFSET = 4
bos_token = "<s>"
pad_token = "<pad>"
eos_token = "<|endoftext|>"
unk_token = "<unk>"

data_dir = Path("gpt2_tokenizer")
tok_v2_dir = data_dir / "v2"

from fairseq.models.transformer_lm import TransformerLanguageModel
root_dir = Path("/checkpoint/dpf/models/lm-1.3B-gpt2tok-fixdata/")
model = TransformerLanguageModel.from_pretrained(root_dir, "checkpoint_last_consolidated.pt", bpe="gpt2", gpt2_encoder_json=f"{root_dir}/vocab.json", gpt2_vocab_bpe=f"{root_dir}/merges.txt")

with open(root_dir / "vocab.json", "r") as fi:
    vocab_json = json.load(fi)

for token, id in vocab_json.items():
    id = model.task.dictionary.indices[str(id)]
    vocab_json[token] = id

# vocab_json[bos_token] = 0
# vocab_json[pad_token] = 1
# vocab_json[eos_token] = 2
# vocab_json[unk_token] = 3

with open(tok_v2_dir / "vocab.json", "w") as fi:
    json.dump(vocab_json, fi)

src_merges_path = root_dir / "merges.txt"
tgt_merges_path = tok_v2_dir / "merges.txt"

shutil.copyfile(src_merges_path, tgt_merges_path)
