from typing import List

TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SPACES8 = " " * 8

COMMENT_DELIMITERS = ('"', "'", "\n")

def split_function_into_docstring_infill_prompt(code: str, docstring_text: str = None) -> List[str]:
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
        prompt_suffix = "{TRIPLE_QUOTE}\n{body_pre_mask}"

    return [prompt_prefix, prompt_suffix]

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

