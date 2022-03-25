""" author: Armen Aghajanyan """
from copy import deepcopy
from operator import index
from os import remove
import astunparse
import ast

from pprint import pprint

class TypeHintKeepOnlyTargeted(ast.NodeTransformer):
    def __init__(self, arg_types, matching_function, remove_type_imports=False):
        # remove_type_imports is False to match TypeWriter paper
        for arg_type in arg_types:
            assert arg_type in ['return', 'argument']
        self.arg_types = arg_types
        self.remove_type_imports = remove_type_imports
        self.matching_function = matching_function
        self.imports = []
        self.matches = []

    def guard(self, callback):
        node = callback(preserve_in_original=(is_target_node==self.preserve_other_types))
        is_target_node = self.iter_index == self.transform_at_index
        if is_target_node:
            self.guard_node = node
            self.guard_value = astunparse.unparse(self.guard_node)

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
        return node

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

def keep_only_return(source, target_line_number, ):
    # TODO: here
    pass


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


def create_return_example(source: str, lineno: int, return_type: str, imports_and_function=True):
    def match_with_type(function, returns):
        return astunparse.unparse(returns).strip() == return_type
    def match_with_line_and_type(function, returns):
        return match_with_type(function, returns) and lineno == function.lineno
    processor = TypeHintKeepOnlyTargeted(['return'], match_with_line_and_type)
    parsed_source = ast.parse(source)
    # remove the type annotations, except for the target
    processor.visit(parsed_source)
    assert len(processor.matches) == 1
    
    if imports_and_function:
        extra_left = [astunparse.unparse(node).strip() for node in processor.imports]
        to_split = astunparse.unparse(processor.matches[0]['node'])
    else:
        extra_left = []
        to_split = astunparse.unparse(parsed_source)
    pairs = list(derive_prefix_suffix(to_split, f" -> {return_type}"))
    assert len(pairs) == 1
    left, right = pairs[0]
    return {
        'extra_left': extra_left,
        'left': left + ' -> ',
        'right': right,
        'return_type': return_type,
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

