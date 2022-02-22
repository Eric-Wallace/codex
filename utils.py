from typing import List, Tuple

TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SPACES8 = " " * 8
EOF = "<|/ file |>"

COMMENT_DELIMITERS = ('"', "'", "\n")

def build_docstring_infill_prompt(code: str, docstring_text: str = None) -> List[str]:
    """Splits the function into a prompt prefix and suffix for the code -> docstring infilling task.

    Args:
        code: text of the function to split
        docstring_text: exact text of the docstring if it's already in the code string and should be stripped out

    Returns:
        list of len 2, splitting code into the part before and after the docstring
    """
    assert code.startswith("def") or code.startswith("async def"), "Must be a function definition"

    if docstring_text is not None:
        # note that we will infill using whatever docstring quote used originally in the function (could be """, ''', #, ', ")
        prompt_prefix = code[:code.index(docstring_text)]
        prompt_suffix = code[code.index(docstring_text) + len(docstring_text):]
    else:
        function_def = code[:code.index(":") + 1]
        body = code[code.index(":") + 1:]
        prompt_prefix = f"{function_def}\n{SPACES4}{TRIPLE_QUOTE}"
        prompt_suffix = "{TRIPLE_QUOTE}\n{body}"

    prompt_suffix += f"\n{EOF}"

    return [prompt_prefix, prompt_suffix]

def build_random_chunk_repair_infill_prompt(code: str) -> List[str]:
    """Creates a prompt to test repair by randomly masking a chunk of code.
    """
    pass

def build_systematic_infill_prompt(original_prompt: str, code: str, num_before: int, num_after: int) -> Tuple[List[str], str]:
    """Creates a prompt with given number of lines before and after to test infill systematically.
    
    Returns:
        prompt_parts (List[str]): list of len 2 [prefix, suffix]
        missing_lines (str): missing part to infill"""
    code_lines = code.split("\n")
    assert num_before + num_after < len(code_lines)
    assert original_prompt[-1] == "\n"
    prefix = "\n".join(code_lines[:num_before])
    suffix = "\n".join(code_lines[len(code_lines) - num_after:])
    missing_lines = "\n".join(code_lines[num_before:len(code_lines) - num_after])

    assert len(prefix.split("\n")) == num_before or (num_before == 0 and len(prefix) == 0)
    assert len(suffix.split("\n")) == num_after or (num_after == 0 and len(suffix) == 0)

    prompt_prefix = original_prompt + prefix
    if not prompt_prefix.endswith("\n"):
        prompt_prefix += "\n"

    return [prompt_prefix, suffix], missing_lines

def truncate_docstring_infill(infill: str) -> str:
    """Truncates an infill to the docstring text, removing extraneous generation output (e.g. additional functions).

    Note: assumes that there's no ' or " within the valid docstring
    """
    # remove leading whitespace
    infill = infill.strip()
    # try to figure out where the end of the comment is
    for delim in COMMENT_DELIMITERS: 
        if delim in infill:
            infill = infill[:infill.index(delim)]
    # remove trailing whitespace
    infill = infill.strip()
    return infill