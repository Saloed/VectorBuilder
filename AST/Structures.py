from collections import namedtuple

Nodes = namedtuple('Nodes', ['root_node', 'all_nodes', 'non_leafs'])


class Author:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __eq__(self, other):
        return self.name == other.name or self.email == other.email

    def __hash__(self):
        return 17 * hash(self.name) + 31 * hash(self.email)

    def __str__(self):
        return '{} <{}>'.format(self.name, self.email)

    def __repr__(self):
        return '{} <{}>'.format(self.name, self.email)


class Token:
    def __init__(self, token_type, parent, is_leaf, pos=0, start_line=None, end_line=None, author=None):
        self.token_type = token_type
        self.parent = parent
        self.children = []
        self.is_leaf = is_leaf
        self.pos = pos
        self.left_rate = 0
        self.right_rate = 0
        self.leaf_num = 0
        self.children_num = 0
        self.index = None
        self.start_line = start_line
        self.end_line = end_line
        self.author = author

    def __str__(self):
        return str(self.token_type) + '_' + str(self.index)

    def __repr__(self):
        return self.__str__()
