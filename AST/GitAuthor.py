from git import *
from os import walk
from collections import Counter, namedtuple
import _pickle as P
from AST.Tokenizer import parser_init, build_ast, print_ast, divide_by_methods, Nodes

MAGIC_CONST = 0.68

DataSet = namedtuple('DataSet', ['methods_with_authors', 'all_authors'])


def parse_directory(dir, f: list):
    for (dirpath, dirnames, filenames) in walk(dir):
        full_file_names = [dirpath + '/' + filename for filename in filenames if
                           filename.endswith('.java')]
        f.extend(full_file_names)
    return f


def _find_author(method: Nodes, authors: dict) -> Nodes:
    m_root = method.root_node
    lines = range(m_root.start_line, m_root.end_line + 1)
    lines_amount = lines.stop - lines.start
    m_authors = [str] * lines_amount
    for l in lines:
        m_authors[l - lines.start] = authors[l]
    cnt = Counter(m_authors)
    author, lns = cnt.most_common(1)[0]
    if lns / lines_amount > MAGIC_CONST:
        m_root.author = author
    return method


def _associate_authors(file, authors: dict, parser):
    ast = build_ast(file, parser)
    methods = divide_by_methods(ast)
    for m in methods:
        _find_author(m, authors)
    return [method for method in methods if method.root_node.author is not None]


def count_authors(data: DataSet):
    authors = {}
    for d in data.methods_with_authors:
        if d.root_node.author not in authors:
            authors[d.root_node.author] = 0
        authors[d.root_node.author] += 1
    return authors


def get_single_author_data(repo_path):
    data = get_repo_methods_with_authors(repo_path)
    authors = count_authors(data)
    max_cnt = 0
    author = None
    for a, cnt in authors.items():
        if cnt > max_cnt:
            max_cnt = cnt
            author = a
    return DataSet([d for d in data.methods_with_authors if d.root_node.author == author], [author])


def get_repo_methods_with_authors(repo_path) -> DataSet:
    parser = parser_init()
    repo = Repo(repo_path)
    files = parse_directory(repo_path, [])
    methods_with_authors = []
    for i, file in enumerate(files):
        file_in_repo = file.replace(repo_path, "")
        print(file_in_repo + ' {}/{}'.format(i, len(files)))
        blames = repo.blame_incremental('HEAD', file_in_repo)
        authors = {}
        for blame in blames:
            lines = blame.linenos
            author = blame.commit.author
            for l in lines:
                authors[l] = author
        methods_with_authors.extend(_associate_authors(file, authors, parser))
    all_authors = []
    for met in methods_with_authors:
        author = met.root_node.author
        if author not in all_authors:
            all_authors.append(author)
    return DataSet(methods_with_authors, all_authors)


if __name__ == '__main__':
    repo_path = '../Dataset/OneAuthorProjects/AndEngine/'
    good_methods = get_repo_methods_with_authors(repo_path)
    with open('test_data', 'wb')as f:
        P.dump(good_methods, f)
