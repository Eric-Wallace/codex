""" original author: Armen Aghajanyan """
from copy import deepcopy
from operator import index
from os import remove
import astunparse
import ast
import libcst as cst
from typing import Optional
import re

arrow_re = re.compile("\s*->\s*")

class TypeHintKeepOnlyTargetedFormatPreserving(cst.CSTTransformer):
    # based on https://stackoverflow.com/questions/42733877/remove-type-hints-in-python-source-programmatically
    def __init__(self, arg_types, matching_function, remove_type_imports=False):
        # remove_type_imports is False to match TypeWriter paper
        for arg_type in arg_types:
            assert arg_type in ['return', 'argument']
        self.arg_types = arg_types
        self.remove_type_imports = remove_type_imports
        if remove_type_imports:
            raise NotImplementedError()
        self.matching_function = matching_function
        self.imports = []
        self.matches = []

    def leave_FunctionDef(self, node, updated_node):
        if 'return' in self.arg_types and node.returns is not None and self.matching_function(function=node, returns=node.returns):
            is_target = True
        else:
            is_target = False
            updated_node = updated_node.deep_remove(updated_node.returns)

        if node.params.params:
            args = []
            for arg in node.params.params:
                if 'argument' in self.arg_types and arg is not None and self.matching_function(function=node, arg=arg):
                    self.matches.append({'node': node, 'arg': arg})
                else:
                    arg = arg.deep_remove(arg.annotation)
                args.append(arg)
            updated_node = updated_node.with_changes(params=node.params.with_changes(params=args))

        if is_target:
            self.matches.append({'original_node': node, 'node': updated_node, 'returns': updated_node.returns})
        return updated_node

    def leave_AnnAssign(self, node, updated_node):
        new_node = cst.Assign([cst.AssignTarget(
            target=updated_node.target,
            whitespace_before_equal=cst.SimpleWhitespace(' '),
            whitespace_after_equal=cst.SimpleWhitespace(' '),
        )], updated_node.value)
        return new_node

    def leave_Import(self, node, updated_node):
        self.imports.append(node)
        return updated_node

    def leave_ImportFrom(self, node, updated_node):
        self.imports.append(node)
        return updated_node

class TypeHintKeepOnlyTargeted(ast.NodeTransformer):
    # based on https://stackoverflow.com/questions/42733877/remove-type-hints-in-python-source-programmatically
    def __init__(self, arg_types, matching_function, remove_type_imports=False):
        # remove_type_imports is False to match TypeWriter paper
        for arg_type in arg_types:
            assert arg_type in ['return', 'argument']
        self.arg_types = arg_types
        self.remove_type_imports = remove_type_imports
        self.matching_function = matching_function
        self.imports = []
        self.matches = []

    def visit_FunctionDef(self, node):
        if 'return' in self.arg_types and node.returns is not None and self.matching_function(function=node, returns=node.returns):
            self.matches.append({'node': node, 'returns': node.returns})
        else:
            node.returns = None

        if node.args.args:
            for arg in node.args.args:
                if 'argument' in self.arg_types and arg is not None and self.matching_function(function=node, arg=arg):
                    self.matches.append({'node': node, 'arg': arg})
                else:
                    arg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.Assign([node.target], node.value)

    def visit_Import(self, node):
        self.imports.append(node)
        if self.remove_type_imports:
            node.names = [n for n in node.names if n.name != 'typing']
            return node if node.names else None
        return node

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        if self.remove_type_imports and node.module == 'typing':
            return None
        return node


class TypeHintRemover(ast.NodeTransformer):
    def __init__(self, transform_at_index, preserve_other_types=True, remove_type_imports=True):
        self.transform_at_index = transform_at_index
        self.iter_index = -1
        self.guard_value = None
        self.preserve_other_types = preserve_other_types
        self.remove_type_imports = remove_type_imports

    def guard(self, callback):
        self.iter_index += 1
        if self.transform_at_index < 0:
            return
        is_target_node = self.iter_index == self.transform_at_index
        node = callback(preserve_in_original=(is_target_node==self.preserve_other_types))
        if is_target_node:
            self.guard_node = node
            self.guard_value = astunparse.unparse(self.guard_node)

    def visit_FunctionDef(self, node):
        def node_empty(preserve_in_original):
            if preserve_in_original:
                to_ret = deepcopy(node.returns)
            else:
                to_ret = node.returns
            node.returns = None
            return to_ret

        if node.returns is not None:
            self.guard(node_empty)

        if node.args.args:
            for arg in node.args.args:
                def node_empty(preserve_in_original):
                    if preserve_in_original:
                        to_ret = deepcopy(arg.annotation)
                    else:
                        to_ret = arg.annotation
                    arg.annotation = None
                    return to_ret
                if arg.annotation is not None:
                    self.guard(node_empty)
        return node

    def visit_Import(self, node):
        if self.remove_type_imports:
            node.names = [n for n in node.names if n.name != 'typing']
            return node if node.names else None
        return node

    def visit_ImportFrom(self, node):
        if self.remove_type_imports and node.module == 'typing':
            return None
        return node


def derive_prefix_suffix(original_source: str, removed_value: str):
    index = original_source.find(removed_value, 0)
    while index >= 0:
        yield original_source[:index], original_source[index + len(removed_value):]
        index = original_source.find(removed_value, index + 1)

def normalize_type(type_, requires_parse=True) -> str:
    # type_: str if requires_parse; else AST
    if requires_parse:
        try:
            parsed = ast.parse(type_)
        except Exception as e:
            print(f"could not parse type {type_}")
            print(e)
            return None
    else:
        parsed = type_
    return astunparse.unparse(parsed).strip()

def create_return_example(source: str, lineno: int, return_type: Optional[str], imports_and_function=True):
    # pass None for return_type if the type is unknown to not require a type match (@@UNK@@ in the typewriter data)
    wrapper = cst.MetadataWrapper(cst.parse_module(source))
    position = wrapper.resolve(cst.metadata.PositionProvider)
    parsed_source = wrapper.module
    def match_with_line_and_type(function, returns):
        function_lineno = position[function].start.line
        return_lineno = position[returns].start.line
        this_return_type = parsed_source.code_for_node(returns.annotation)
        matches_type = (return_type is None) or normalize_type(this_return_type, requires_parse=True) == normalize_type(return_type, requires_parse=True)
        matches_line = (lineno == return_lineno) or (lineno == function_lineno) or (lineno == function_lineno-2) or (lineno == function_lineno-1)
        return matches_type and matches_line
    processor = TypeHintKeepOnlyTargetedFormatPreserving(['return'], match_with_line_and_type)
    # remove the type annotations, except for the target
    try:
        transformed_parsed_source = parsed_source.visit(processor)
    except Exception as e:
        # visit has trouble with e.g. assignments where the value is an unimported class
        print(e)
        return None

    if len(processor.matches) != 1:
        print(f"{len(processor.matches)} matches found!")
        print(f"return_type: {return_type}")
        print('\n'.join(source.splitlines()[lineno-1:lineno+2]))
        return None

    return_type_from_source = transformed_parsed_source.code_for_node(processor.matches[0]['returns'].annotation)

    return_type_from_source = normalize_type(return_type_from_source)
    
    if imports_and_function:
        extra_left = [transformed_parsed_source.code_for_node(node) for node in processor.imports]
        try:
            to_split = transformed_parsed_source.code_for_node(processor.matches[0]['node'])
        except Exception as e:
            print(e)
            return None
    else:
        extra_left = []
        try:
            to_split = transformed_parsed_source.code
        except Exception as e:
            # .code has trouble with e.g. assignments of boolean values created by an equality test
            print(e)
            return None
    
    to_split = arrow_re.sub(" -> ", to_split)
    pairs = list(derive_prefix_suffix(to_split, f" -> {return_type_from_source}"))
    if len(pairs) != 1:
        return None
    assert len(pairs) == 1
    left, right = pairs[0]
    return {
        'extra_left': extra_left,
        'left': left + ' -> ',
        'right': right,
        'return_type_from_source': return_type_from_source,
    }


if __name__ == "__main__":
    source = """
import typing
from typing import Dict, T, Callable
from typing import List

def foo(bar: Dict[T, List[T]],
        baz: Callable[[T], int] = lambda x: (x+3)/7,
        **kwargs):
    a: int = True
    pass
"""

    # with open('/tmp/source.py', 'r') as f:
    #     source = f.read()

    parsed_source = ast.parse(source)
    clean_source = astunparse.unparse(TypeHintRemover(len(source)).visit(ast.parse(source)))
    unique = set()
    for i in range(len(source)):
        transformer = TypeHintRemover(i)
        transformed_ast = transformer.visit(parsed_source)
        transformed = astunparse.unparse(transformed_ast)
        unique.add((transformed, transformer.guard_value))
    infills = []
    for (removed_source, removed_value) in unique:
        print(removed_source)
        print(removed_value)
        if removed_value is None:
            continue
        for left, right in derive_prefix_suffix(clean_source, removed_value.strip()):
            print("left:", left)
            print("right:", right)
            print("removed:", removed_value)
            print()
            infills.append((left,right,removed_value))
        print()
    #pprint(infills)
