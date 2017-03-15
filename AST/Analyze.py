import subprocess

import sys
import _pickle as P
import time
from py4j.java_gateway import GatewayParameters, JavaGateway, get_field

from AST.Tokenizer import _parse_tree, tree_to_list, Nodes, Author


def analyzer_init(port=None):
    if port is None:
        port = 25537
    process = subprocess.Popen(
        ["java", "-jar", "/home/sobol/PycharmProjects/VectorBuilder/AST/AuthorAnalyzer.jar", str(port)],
        stdout=sys.stdout, stderr=sys.stderr)
    time.sleep(1)
    parameters = GatewayParameters(port=port)
    gateway = JavaGateway(gateway_parameters=parameters)
    analyzer = gateway.entry_point.getMain()
    return analyzer, gateway, process


def process_ast(ast):
    root_node = _parse_tree(ast)
    author = get_field(ast, 'author')
    if author is not None:
        author = Author(get_field(author, 'name'), get_field(author, 'email'))
        root_node.author = author
    all_nodes = tree_to_list(root_node)
    non_leafs = [node for node in all_nodes if not node.is_leaf]
    return Nodes(root_node, all_nodes, non_leafs)


if __name__ == '__main__':
    repo_path = "Dataset/OneAuthorProjects/distributedlog/"
    analyzer, gateway, process = analyzer_init()
    analyzer_data = analyzer.analyzeRepo(repo_path)
    print('End data generation')
    data = []
    for i, ast in enumerate(analyzer_data):
        if ast is not None:
            d = process_ast(ast)
            data.append(d)
        print('Constructed {} / {}'.format(i, len(analyzer_data)))
    with open('Dataset/OneAuthorProjects/distributedlog_file', 'wb') as f:
        P.dump(data, f)
    gateway.shutdown()
    process.terminate()
