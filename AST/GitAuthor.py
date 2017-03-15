from multiprocessing.pool import Pool
import logging
from git import *
from os import walk
from collections import Counter, namedtuple
import _pickle as P
from AST.Tokenizer import *
from AST.Tokenizer import parser_init, get_psi_text

MAGIC_CONST = 0.68

DataSet = namedtuple('DataSet', ['methods_with_authors', 'all_authors'])


def parse_directory(directory, f: list):
    for (dir_path, dir_names, file_names) in walk(directory):
        full_file_names = [dir_path + '/' + filename for filename in file_names if
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


def _process_files(repo, files, _parser):
    parser, gateway, process = _parser
    methods_with_authors = []
    for i, file in enumerate(files):
        file_in_repo, file = file
        logging.info(file_in_repo + ' {}/{}'.format(i, len(files)))
        blames = repo.blame_incremental('HEAD', file_in_repo)
        authors = {}
        for blame in blames:
            lines = blame.linenos
            author = blame.commit.author
            for l in lines:
                authors[l] = author
        methods_with_authors.extend(_associate_authors(file, authors, parser))
    gateway.shutdown()
    process.terminate()
    return methods_with_authors


def get_repo_methods_with_authors(repo_path) -> DataSet:
    logging.basicConfig(level=logging.INFO)
    repo = Repo(repo_path)
    files = parse_directory(repo_path, [])
    pool_size = 8
    chunk_size = len(files) // pool_size
    base_port_address = 25347
    parsers = [parser_init(base_port_address + i) for i in range(pool_size)]
    files = [(file.replace(repo_path, ""), file) for file in files]
    files = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    pool = Pool(pool_size)
    results = [pool.apply_async(_process_files, (repo, files[i], parsers[i])) for i in range(pool_size)]
    methods_with_authors = []
    for res in results:
        methods_with_authors.extend(res.get())
    all_authors = []
    for met in methods_with_authors:
        author = met.root_node.author
        if author not in all_authors:
            all_authors.append(author)
    return DataSet(methods_with_authors, all_authors)


def build_psi_text(data_set_dir):
    parser, gateway, process = parser_init(25537)
    files = parse_directory(data_set_dir, [])
    psi_text_file = []
    for file in files:
        try:
            print(file)
            psi_text = get_psi_text(file, parser)
            psi_text_file.append(psi_text)
        except Exception as ex:
            print(ex)
            continue
    print('end ast building')
    delimiter = 'WHITE_SPACE ' * 5 + '\n'
    text = delimiter.join(psi_text_file)
    gateway.shutdown()
    process.terminate()
    with open('/home/sobol/PycharmProjects/VectorBuilder/Dataset/psi_text_1.data', 'w') as text_file:
        text_file.write(text)


if __name__ == '__main__':
    repo_path = '../Dataset/OneAuthorProjects/AndEngine/'
    good_methods = get_repo_methods_with_authors(repo_path)
    with open('test_data', 'wb')as f:
        P.dump(good_methods, f)
