from git import *


def test(path_to_repo, filename, start_line, end_line):
    repo = Repo(path_to_repo)
    repo.blame('HEAD', filename, )
