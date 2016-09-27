import time
from antlr4 import *
from AST.JANTLR.Java8Lexer import Java8Lexer
from AST.JANTLR.Java8Parser import Java8Parser
from antlr4.tree.Trees import Trees
from Utils.Wrappers import timing


@timing
def doparsing(filename):
    input = FileStream(filename)
    lexer = Java8Lexer(input)
    stream = CommonTokenStream(lexer)
    parser = Java8Parser(stream)
    tree = parser.compilationUnit()
    print(Trees.toStringTree(tree, None, parser))
    return tree


def main():
    tree = doparsing('../Dataset/java_files/Animation.java')
    pass


if __name__ == '__main__':
    main()
