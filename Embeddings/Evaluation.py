import theano.tensor as T
from theano import function
import numpy as np
from AST.Sampler import PreparedAST
from AST.Token import Token
from Embeddings.Parameters import *
from TBCNN.NetworkParams import Network
from Utils.Wrappers import timing


class EvaluationSet:
    def __init__(self, sample: PreparedAST):
        self.sample = sample
        self.back_prop = None
        self.validation = None


# @timing
def process_network(eval_set: EvaluationSet, params: Parameters, alpha, decay, is_validation):
    change = np.random.randint(0, len(params.embeddings) - 1)
    secret = params.embeddings[change].eval()
    if not is_validation:
        train_error = eval_set.back_prop(secret, alpha, decay)
    else:
        train_error = eval_set.validation(secret)
    return train_error / eval_set.sample.ast_len, train_error
