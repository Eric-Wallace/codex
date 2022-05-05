import os
from transformers import PreTrainedTokenizerFast
from pathlib import Path
from tokenizers import pre_tokenizers, processors, ByteLevelBPETokenizer, Regex

data_dir = Path("tokenizer")

tok_v1_dir = data_dir / "v1"
tok_v2_dir = data_dir / "v2"
trfs_tok_dir = data_dir / "v2_trfs"

BOS = "<|endoftext|>"
EOM = "<|endofmask|>"

FAIRSEQ_EOS_ID = 2
# fairseq reserves the initial items in the vocab for eos, bos, pad, and unk (not necessarily in that order)
TOKEN_OFFSET = 4
SPECIAL_TOKENS = [f"<|mask:{i}|>" for i in range(256)] + [EOM]

backend_tokenizer_v2 = ByteLevelBPETokenizer.from_file(
    os.path.join(tok_v2_dir, "vocab.json"),
    os.path.join(tok_v2_dir, "merges.txt")
)
backend_tokenizer_v2.add_special_tokens(SPECIAL_TOKENS)
backend_tokenizer_v2.pre_tokenizer = pre_tokenizers.Sequence(
    [
    pre_tokenizers.Split(Regex(r"[\r\n]+"), "isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ]
)
backend_tokenizer_v2.post_processor = processors.TemplateProcessing(
    single=f"{BOS} $0",
    pair=f"{BOS} $A {BOS}:1 $B:1",
    special_tokens=[(BOS, 2)],
)

# wrap and save tokenizer
new_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend_tokenizer_v2,
    model_max_length=2048,
    # padding_side="Set me if you want",
    # truncation_side="Set me if you want",
    # model_input_names="Set me if you want",
    # bos_token="Set me if you want",
    bos_token=BOS,
    # unk_token="Set me if you want",
    # sep_token="Set me if you want",
    # pad_token="Set me if you want",
    # cls_token="Set me if you want",
    # mask_token="Set me if you want",
)


#completion = generate("def count_words(filename):\n", do_sample=True, top_p=0.95, temperature=0.2, max_length=128)

new_tokenizer.save_pretrained(trfs_tok_dir)
