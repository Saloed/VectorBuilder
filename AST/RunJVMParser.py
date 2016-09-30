import os
import subprocess

if __name__ == '__main__':
    subprocess.run(["java", "-jar", os.getcwd() + "/ASTParser.jar"])
