import theano.tensor as T
from theano import function

from AST.Token import Token
from Embeddings.Parameters import *
from TBCNN.NetworkParams import Network
from Utils.Wrappers import timing


class EvaluationSet:
    def __init__(self, back_prop, training_token, ast_len):
        self.back_prop = back_prop
        self.training_token = training_token
        self.ast_len = ast_len


def evaluate(back_prop, training_token: Token,
             params: Parameters, alpha, decay):
    error = back_prop(alpha, decay)

    return error


# @timing
def process_network(eval_set: EvaluationSet, params, alpha, decay):
    train_error = evaluate(eval_set.back_prop,
                           eval_set.training_token, params, alpha, decay)
    return train_error / eval_set.ast_len, train_error
