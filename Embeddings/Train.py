from AST.Tokenizer import build_ast
from Embeddings.InitEmbeddings import initialize

params = initialize()

ast = build_ast('../Dataset/test_0.java')
