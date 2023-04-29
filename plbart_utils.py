# extracted from https://raw.githubusercontent.com/wasiahmad/PLBART/main/data/github/preprocessing/src/code_tokenizer.py

import argparse
import logging
import os
import random
import re
import sys
import io
import tokenize

from io import BytesIO

TOK_NO_SPACE_BEFORE = {',', ';'}

logging.basicConfig(level=logging.INFO)

PYTHON_TOKEN2CHAR = {
    'STOKEN0': '#',
    'STOKEN1': "\\n",
    'STOKEN2': '"""',
    'STOKEN3': "'''"
}

PYTHON_CHAR2TOKEN = {
    '#': ' STOKEN0 ',
    "\\n": ' STOKEN1 ',
    '"""': ' STOKEN2 ',
    "'''": ' STOKEN3 '
}


class ind_iter(object):
    def __init__(self, len):
        self.i = 0
        self.len = len

    def next(self):
        self.i += 1
        if self.i > (self.len - 1):
            raise StopIteration

    def prev(self):
        self.i -= 1
        if self.i < 0:
            raise StopIteration


def replace_tokens(tok, dictionary):
    for char, special_token in dictionary.items():
        tok = tok.replace(char, special_token)
    return tok


def replace_general_string_tok(tok):
    return (
        tok.replace(" ", " ▁ ")
            .replace("\n", " STRNEWLINE ")
            .replace("\t", " TABSYMBOL ")
    )


def process_string(tok, char2tok, tok2char, is_comment, do_whole_processing=True):
    from sacrebleu import tokenize_v14_international
    if not (do_whole_processing or is_comment):
        return tok.replace("\n", "\\n").replace("\r", "")

    if is_comment:
        tok = re.sub(" +", " ", tok)
        tok = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", tok)
        if len(re.sub(r"\W", "", tok)) < 2:
            return ""
    tok = replace_general_string_tok(tok)
    tok = replace_tokens(tok, char2tok)
    if tok.strip().startswith("STOKEN00"):
        if " STRNEWLINE " in tok:
            tok = tok.replace(" STRNEWLINE ", " ENDCOM", 1)
        else:
            tok += " ENDCOM"
    if not do_whole_processing:
        tok = replace_tokens(
            tok, {f" {key} ": value for key, value in tok2char.items()}
        )
        tok = (
            tok.replace(" ▁ ", " ")
                .replace(" TABSYMBOL ", "\t")
                .replace("\\r", "")
                .replace(" STRNEWLINE ", "\\n")
        )
        return tok

    tok = re.sub(" +", " ", tok)
    tok = tokenize_v14_international(tok)
    tok = re.sub(" +", " ", tok)
    tok = tok.replace("\r", "")
    for special_token, char in tok2char.items():
        tok = tok.replace(special_token, char)
    if tok[0].isalpha():
        # for special strings, (e.g. L "s" we should remove the space after L)
        tok = tok.replace(f"{tok[0]} ", tok[0])
    return tok


def tokenize_python(code, keep_comments=False, process_strings=True):
    assert isinstance(code, str)
    code = code.replace(r"\r", "")
    code = code.replace("\r", "")
    tokens = []

    try:
        iterator = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
    except SyntaxError as excep:
        raise SyntaxError(excep)

    removed_docstr = 0
    while True:
        try:
            toktype, tok, _, _, line = next(iterator)
        except (
                tokenize.TokenError,
                IndentationError,
                SyntaxError,
                UnicodeDecodeError,
        ) as e:
            raise ValueError(
                f'Impossible to parse tokens because of incorrect source code "{e}" ...'
            )
        except StopIteration:
            raise Exception(f"End of iterator before ENDMARKER token.")

        if toktype == tokenize.ENCODING or toktype == tokenize.NL:
            continue

        elif toktype == tokenize.NEWLINE:
            if removed_docstr == 1:
                removed_docstr = 0
                continue
            tokens.append("NEW_LINE")

        elif toktype == tokenize.COMMENT:
            if keep_comments:
                com = process_string(
                    tok,
                    PYTHON_CHAR2TOKEN,
                    PYTHON_TOKEN2CHAR,
                    True,
                    do_whole_processing=process_strings,
                )
                if len(com) > 0:
                    tokens.append(com)
            else:
                continue

        elif toktype == tokenize.STRING:
            if tok == line.strip():  # docstring
                if not keep_comments:
                    removed_docstr = 1
                    continue
                else:
                    coms = process_string(
                        tok,
                        PYTHON_CHAR2TOKEN,
                        PYTHON_TOKEN2CHAR,
                        False,
                        do_whole_processing=process_strings,
                    )
                    if len(coms) > 0:
                        tokens.append(coms)
                    else:
                        removed_docstr = 1
            else:
                tokens.append(
                    process_string(
                        tok,
                        PYTHON_CHAR2TOKEN,
                        PYTHON_TOKEN2CHAR,
                        False,
                        do_whole_processing=process_strings,
                    )
                )

        elif toktype == tokenize.INDENT:
            tokens.append("INDENT")

        elif toktype == tokenize.DEDENT:
            # empty block
            if tokens[-1] == "INDENT":
                tokens = tokens[:-1]
            else:
                tokens.append("DEDENT")

        elif toktype == tokenize.ENDMARKER:
            tokens.append("ENDMARKER")
            break

        else:
            tokens.append(tok)

    assert tokens[-1] == "ENDMARKER", "Error, no end marker"
    return tokens[:-1]


def detokenize_code(code, start_tabs=0):
    # replace recreate lines with \n and appropriate indent / dedent
    # removing indent/ dedent tokens
    assert isinstance(code, str) or isinstance(code, list)
    if isinstance(code, list):
        code = " ".join(code)
    code = code.replace("ENDCOM", "NEW_LINE")
    code = code.replace("▁", "SPACETOKEN")
    lines = code.split("NEW_LINE")
    lines = [line for line in lines if line.strip()]
    tabs = "    " * start_tabs
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("INDENT "):
            tabs += "    "
            line = line.replace("INDENT ", tabs)
        elif line.startswith("DEDENT"):
            number_dedent = line.count("DEDENT")
            tabs = tabs[4 * number_dedent:]
            line = line.replace("DEDENT", "")
            line = line.strip()
            line = tabs + line
        elif line == "DEDENT":
            line = ""
        else:
            line = tabs + line
        lines[i] = line
    untok_s = "\n".join(lines)
    # find string and comment with parser and detokenize string correctly
    try:
        for toktype, tok, _, _, line in tokenize.tokenize(
                BytesIO(untok_s.encode("utf-8")).readline
        ):
            if toktype == tokenize.STRING or toktype == tokenize.COMMENT:
                tok_ = (
                    tok.replace("STRNEWLINE", "\n")
                        .replace("TABSYMBOL", "\t")
                        .replace(" ", "")
                        .replace("SPACETOKEN", " ")
                )
                untok_s = untok_s.replace(tok, tok_)
    except KeyboardInterrupt:
        raise
    except:
        # TODO raise ValueError(f'Invalid python function \n {code}\n')
        pass
    # detokenize imports
    untok_s = (
        untok_s.replace(". ", ".")
            .replace(" .", ".")
            .replace("import.", "import .")
            .replace("from.", "from .")
    )
    # special strings
    string_modifiers = ["r", "u", "f", "rf", "fr", "b", "rb", "br"]
    for modifier in string_modifiers + [s.upper() for s in string_modifiers]:
        untok_s = untok_s.replace(f" {modifier} '", f" {modifier}'").replace(
            f' {modifier} "', f' {modifier}"'
        )
    untok_s = untok_s.replace("> >", ">>").replace("< <", "<<")
    return untok_s, len(tabs) // 4


def extract_functions_python_with_docstring(function):
    ds = re.findall(
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ]['][']['].*?['][']['][ ][N][E][W][_][L][I][N][E]|[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ][\"][\"][\"].*?[\"][\"][\"][ ][N][E][W][_][L][I][N][E]",
        function, re.DOTALL)
    if len(ds) > 0:
        for d in ds:
            function = function.replace(d[18:-9], '')
        coms = ' '.join([d[18:-9] for d in ds])
        inline_coms = re.findall('[#].*?[E][N][D][C][O][M]', function)
        for inline_com in inline_coms:
            function = function.replace(inline_com, '')
            coms += ' <INLINE> '
            coms += inline_com
        if len(re.sub(r'\W', '', coms.replace('<INLINE>', '').replace('ENDCOM', ''))) < 5:
            return '', ''
        else:
            return re.sub('\s+', ' ', function), coms
    else:
        return '', ''


def extract_functions_python(s):
    tokens = iter(s.split())
    functions_standalone = []
    functions_class = []
    number_indent = 0
    try:
        token = next(tokens)
    except StopIteration:
        return [], []
    while True:
        try:
            if token == 'def':
                function = ['def']
                while not (token == 'DEDENT' and number_indent == 0):
                    token = next(tokens)
                    if token == 'INDENT':
                        number_indent += 1
                    elif token == 'DEDENT':
                        number_indent -= 1
                    function.append(token)
                try:
                    if function[function.index('(') + 1] == 'self':
                        function = filter_functions_python_2_3(
                            ' '.join(function))
                        if function is not None:
                            functions_class.append(function)
                    else:
                        function = filter_functions_python_2_3(
                            ' '.join(function))
                        if function is not None:
                            functions_standalone.append(function)
                except:
                    print(function)
                    token = next(tokens)
            else:
                token = next(tokens)
        except StopIteration:
            break
    return functions_standalone, functions_class


def filter_functions_python_2_3(function):
    if (re.search("print [^(]", function) is None and
            re.search("raise \w+ ,", function) is None and
            re.search("except \w+ ,", function) is None and
            re.search("[^ ]+ = \d+ L", function) is None and
            ". iterkeys ( )" not in function and
            ". itervalues ( )" not in function and
            ". iteritems ( )" not in function and
            "xrange (" not in function and
            "imap (" not in function):
        return function
    else:
        return None


def get_function_name_python(s):
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, str):
        s = s.split()
    return s[s.index('def') + 1]


def indent_lines(lines):
    prefix = ''
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match('CB_COLON|CB_COMA|CB_', line):
            prefix = prefix[2:]
            line = prefix + line
        elif line.endswith('OB_'):
            line = prefix + line
            prefix += '  '
        else:
            line = prefix + line
        lines[i] = line
    untok_s = '\n'.join(lines)
    return untok_s


def get_first_token_before_first_parenthesis(s):
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, str):
        s = s.split()
    return s[s.index('(') - 1]

def find_longest_suffix(xs, ys):
    # find the longest occurring suffix of xs at the earliest position in ys
    for x_start in range(len(xs)):
        x_suff = xs[x_start:]
        if not x_suff:
            continue
        for y_start in range(len(ys)):
            y_end = y_start+len(x_suff)
            y_suff = ys[y_start:y_end]
            if x_suff == y_suff:
                return x_suff, xs[:x_start], (ys[:y_start], ys[y_end:])
    return None
