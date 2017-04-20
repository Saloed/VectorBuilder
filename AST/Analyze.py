import _pickle as P
import sys
from os import walk

from AST.Utility import analyzer_init, process_ast, author_collapse


def get_all_available_tokens() -> list:
    parser, gateway, process = analyzer_init()
    tokens = parser.getAllAvailableTokens()
    result = [str(token) for token in tokens]
    gateway.shutdown()
    process.terminate()
    return result


def build_psi_text(data_set_dirs, result_name):
    analyzer, gateway, process = analyzer_init()
    files = []
    for data_set_dir in data_set_dirs:
        files.extend(parse_directory(data_set_dir, []))
    psi_text_file = []
    for file in files:
        try:
            print(file)
            psi_text = str(analyzer.parsePSIText(file))
            psi_text_file.append(psi_text)
        except Exception as ex:
            print(ex)
            continue
    print('end ast building')
    delimiter = 'WHITE_SPACE ' * 5 + '\n'
    text = delimiter.join(psi_text_file)
    gateway.shutdown()
    process.terminate()
    with open(result_name, 'w') as text_file:
        text_file.write(text)


def parse_directory(directory, f: list):
    for (dir_path, dir_names, file_names) in walk(directory):
        full_file_names = [dir_path + '/' + filename for filename in file_names if
                           filename.endswith('.java')]
        f.extend(full_file_names)
    return f


def process_repo(repo_name):
    repo_path = repo_name + '/'
    analyzer, gateway, process = analyzer_init()
    analyzer_data = analyzer.analyzeRepo(repo_path)
    print('End data generation')
    data = []
    for i, ast in enumerate(analyzer_data):
        if ast is not None:
            d = process_ast(ast)
            data.append(d)
        print('Constructed {} / {}'.format(i, len(analyzer_data)))
    authors = list({m.root_node.author for m in data})
    uauthors = author_collapse(authors)
    data = [([m for m in data if m.root_node.author in ua], ua) for ua in uauthors]
    with open(repo_name + '_file', 'wb') as f:
        P.dump(data, f)
    gateway.shutdown()
    process.terminate()


if __name__ == '__main__':
    build_psi_text(['Dataset/intellij-community'], 'tmp_result.hlam')
    # arg = sys.argv[1]
    # process_repo(arg)
