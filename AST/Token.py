# todo full rebase on dict (remove pos)
class Token:
    token_type = None
    token_index = None
    parent = None
    children = None
    pos = 0
    left_rate = 0
    right_rate = 0
    leaf_num = 0
    children_num = 0

    def __init__(self, token_type, token_index, parent, pos=0):
        self.token_type = token_type
        self.token_index = token_index
        self.parent = parent
        self.children = []
        self.pos = pos
        self._hash = pos

    # redefined for use in dict
    def __hash__(self):
        return self._hash

    def __str__(self):
        return str(self.token_type) + str(self._hash)
