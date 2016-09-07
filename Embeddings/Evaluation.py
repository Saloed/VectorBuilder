import theano.tensor as T
from theano import function

from AST.Token import Token
from Embeddings.Parameters import *
from TBCNN.NetworkParams import Network
from Utils.Wrappers import timing


class EvaluationSet:
    def __init__(self, positive, negative, training_token, ast_len):
        self.positive = positive
        self.negative = negative
        self.training_token = training_token
        self.ast_len = ast_len


x = T.fvector('x')
mse = function(inputs=[x], outputs=T.mul(T.sum(T.mul(x, x)), 0.5))


def evaluate(positive: Network, negative: Network, training_token: Token,
             params: Parameters, alpha, decay, is_validation):
    pos_forward = positive.forward()
    neg_forward = negative.forward()

    target = params.embeddings[training_token.token_index].eval()
    pos_diff = pos_forward - target
    neg_diff = neg_forward - target

    pos_mse = mse(pos_diff)
    neg_mse = mse(neg_diff)

    error = MARGIN + pos_mse - neg_mse

    if error < 0:
        return 0
    if is_validation:
        return error

    positive.back(target, neg_forward, alpha, decay)
    negative.back(target, pos_forward, alpha, decay)

    return error


# @timing
def process_network(eval_set: EvaluationSet, params, alpha, decay, is_validation):
    train_error = 0
    for neg in eval_set.negative:
        train_error += evaluate(eval_set.positive, neg,
                                eval_set.training_token, params, alpha, decay, is_validation)
    return train_error / eval_set.ast_len, train_error
