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
        string = "\n" + shift + "┗ " + self.node_name
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


# def start_jvm_with_parser():
#     sp = subprocess.call(["java", "-jar", "ASTParser.jar"])


def main():
    jvm_process = subprocess.Popen(["java", "-jar", os.getcwd() + "/ASTParser.jar"])
    # jvm_process.start()
    # wait for jvm run
    time.sleep(1)
    filename = '../Dataset/java_files/Animation.java'
    gateway = JavaGateway()
    parser = gateway.entry_point.getParser()
    ast = parser.parseFile(filename)

    jvm_process.kill()

    ast = parse_tree(ast)
    print(ast)


if __name__ == '__main__':
    main()
