from typing import List

TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SPACES8 = " " * 8

def split_function_into_docstring_infill_prompt(code: str, has_docstring: bool) -> List[str]:
    """Splits the function into a prompt prefix and suffix for the code -> docstring infilling task.

    Args:
        code: text of the function to split
        has_docstring: whether the function text already contains a docstring (that should be stripped out) 

    Returns:
        list of len 2, splitting code into the part before and after the docstring
    """
    assert code.startswith("def"), "Must be a function definition"

    if has_docstring:
        if TRIPLE_QUOTE in code:
            comment_quote = TRIPLE_QUOTE
        elif SINGLE_TRIPLE_QUOTE in code:
            comment_quote = SINGLE_TRIPLE_QUOTE
        else:
            raise Exception("no triple or single quote")

        prompt_prefix = code[:code.index(comment_quote) + 3]
        code = code[code.index(comment_quote) + 3:]
        prompt_suffix = code[code.index(comment_quote):]
    else:
        function_def = code[:code.index(":") + 1]
        body = code[code.index(":") + 1:]
        prompt_prefix = f"{function_def}\n{SPACES4}{TRIPLE_QUOTE}"
        prompt_suffix = "{TRIPLE_QUOTE}\n{body_pre_mask}"

    assert prompt_prefix.endswith(TRIPLE_QUOTE)
    assert prompt_suffix.startswith(TRIPLE_QUOTE)
    return [prompt_prefix, prompt_suffix]
