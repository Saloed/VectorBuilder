import os
import time
from py4j.java_gateway import JavaGateway, get_field
from multiprocessing import Process
import subprocess


class Node:
    def __init__(self, node_name, node_index, is_terminal, src_start, src_end):
        self.node_name = node_name
        self.node_index = node_index
        self.is_terminal = is_terminal
        self.src_start = src_start
        self.src_end = src_end
        self.parent = None
        self.children = []

    def shifted_string(self, shift="") -> str:
        string = '\n{}┗ {} [{}:{}]'.format(shift, self.node_name, self.src_start, self.src_end)
        for child in self.children:
            if child.is_terminal:
                string += " " + child.node_name
            else:
                string += child.shifted_string(shift + " ")
        return string

    def __str__(self) -> str:
        return self.shifted_string()


def parse_node(node) -> Node:
    node_name = get_field(node, 'nodeName')
    node_index = get_field(node, 'nodeIndex')
    is_terminal = get_field(node, 'isTerminal')
    src_start = get_field(node, 'sourceStart')
    src_end = get_field(node, 'sourceEnd')
    return Node(node_name, node_index, is_terminal, src_start, src_end)


def parse_tree(root, parent=None) -> Node:
    node = parse_node(root)
    node.parent = parent
    if parent is not None:
        parent.children.append(node)
    children = get_field(root, 'children')
    for child in children:
        parse_tree(child, node)
    return node


useless_tokens = [
    0,
    1,
    2, 3, 4,
    5, 6, 58, 176, 131, 128, 133, 132, 138
]


def collapse_tree(tree: Node) -> Node:
    if not tree.is_terminal:
        if tree.node_index in useless_tokens and tree.parent is not None:
            assert len(tree.children) == 1
            child = tree.children[0]
            tree.parent.children.remove(tree)
            tree.parent.children.append(child)
            child.parent = tree.parent
            collapse_tree(child)
        else:
            for child in tree.children:
                collapse_tree(child)
    return tree


def main():
    filename = '../Dataset/java_files/ActionId.java'
    gateway = JavaGateway()
    parser = gateway.entry_point.getParser()
    ast = parser.parseFile(filename)

    ast = parse_tree(ast)
    print(ast)
    print('____________________________________')
    cl_ast = collapse_tree(ast)
    print(cl_ast)


if __name__ == '__main__':
    main()
