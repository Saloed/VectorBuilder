from AST.Tokenizer import ast_to_list


def construct_from_ast(ast, parameters):
    nodes = ast_to_list(ast)
