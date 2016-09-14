import os
import pickle
import numpy as np
from copy import deepcopy

from AST import Token
from AST.TokenMap import token_map
from AST.Tokenizer import build_ast


class PreparedAST:
    def __init__(self, positive, training_token, training_token_index, ast_len):
        self.positive = positive
        self.training_token = training_token
        self.training_token_index = training_token_index
        self.ast_len = ast_len
        self.eval_set_file = None


def build_asts(dataset_dir):
    base_dir = dataset_dir
    dataset_dir += 'java_files/'
    files = os.listdir(dataset_dir)
    data_ast = []
    for file in files:
        try:
            ast = build_ast(dataset_dir + file)
            data_ast.append(ast)
        except Exception:
            continue
    ast_file = open(base_dir + 'ast_file', mode='wb')
    pickle.dump(data_ast, ast_file)


def prepare_ast(full_ast, training_token_index):
    nodes_with_depth = []

    def compute_depth(node: Token, depth):
        if len(node.children) != 0:
            nodes_with_depth.append((node, depth))
            for child in node.children:
                compute_depth(child, depth + 1)

    compute_depth(full_ast[-1], 0)
    nodes_with_depth.sort(key=lambda tup: tup[1])
    nodes = [deepcopy(node[0]) for node in reversed(nodes_with_depth)]

    class Indexer:
        def __init__(self):
            self.indexer = 0

        def children(self, node: Token, ast=None, parent=None, need_more=True) -> list:
            if ast is None:
                ast = []
                assert training_token_index == self.indexer
            node.parent = parent
            node.pos = self.indexer
            ast.append(node)
            self.indexer += 1
            if need_more:
                for child in node.children:
                    self.children(child, ast, node.pos, need_more=False)
            else:
                node.children = []
            return ast

    return [Indexer().children(node) for node in nodes]


def generate_samples(data, ast_list: list, training_token_index):
    prepared = prepare_ast(data, training_token_index)
    for ast in prepared:
        ast_len = len(ast)
        if ast_len < 3 or ast_len > 25:
            continue
        training_token = ast[training_token_index]

        def rand_token():
            return list(token_map.keys())[np.random.randint(0, len(token_map))]

        assert training_token_index == 0
        # for token in ast[training_token_index + 1:]:
        #     while token.token_type == training_token.token_type:
        #         new_token = rand_token()
        #         token.token_type = new_token
        #         token.token_index = token_map[new_token]
        # def create_negative(token_index):
        #     sample = deepcopy(ast)
        #     current = sample[token_index]
        #     new_token = rand_token()
        #     while current.token_type == new_token:
        #         new_token = rand_token()
        #     current.token_type = new_token
        #     current.token_index = token_map[new_token]
        #     return sample
        #
        # # rand from 1 because root_token_index is 0
        # samples = [create_negative(i) for i in np.random.random_integers(1, len(ast) - 1, size=1)]

        ast_list.append(PreparedAST(ast, training_token, training_token_index, ast_len))
