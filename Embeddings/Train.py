import _pickle as c_pickle
from random import shuffle

import theano

from AST.Sampler import PreparedAST, generate_samples
from Embeddings.Construct import construct
from Embeddings.Evaluation import EvaluationSet, process_network
from Embeddings.InitEmbeddings import initialize
from Embeddings.Parameters import *
from Utils.Wrappers import *

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'

# 0 because of preparing
training_token_index = 0
log_file = open('log.txt', mode='w')
batch_size = 10


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


# @timing
def prepare_net(ast: PreparedAST, params, is_validation):
    back_prop = construct(ast.positive, params, ast.training_token_index, is_validation)

    return EvaluationSet(back_prop, ast.training_token, ast.ast_len)


@timing
def process_batch(batch, params, alpha, decay, is_validation):
    total_t_err = 0
    total_a_err = 0
    samples_size = len(batch)
    for sample in batch:
        eval_set = prepare_net(sample, params, is_validation)
        ept, err = process_network(eval_set, params, alpha, decay)
        total_t_err += ept
        total_a_err += err

    return total_a_err / samples_size, total_t_err / samples_size


# @safe_run
def process_batches(batches, params, alpha, decay, is_validation):
    error_per_ast = 0
    error_per_token = 0
    for i, batch in enumerate(batches):
        epa, ept = process_batch(batch, params, alpha, decay, is_validation)
        message = ['\t\t|\t{}\t|\t{}\t|\t{}'.format(ept, epa, i)]
        fprint(message)
        error_per_ast += epa
        error_per_token += ept
    return error_per_ast, error_per_token


def create_batches(data):
    samples = []
    for ast in data:
        generate_samples(ast, samples, training_token_index)
    data_size = len(samples)
    data_size //= batch_size
    batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(data_size)]
    return batches


# @safe_run
def epoch_step(params, epoch_num, retry_num, prev, batches, train_set_size, decay):
    # shuffle(batches)
    train_set = batches[:train_set_size]
    validation_set = batches[train_set_size:]
    alpha, prev_t_ast, prev_t_token, prev_v_ast, prev_v_token = prev
    fprint(['train set'])
    result = process_batches(train_set, params, alpha, decay, False)
    if result is None:
        return
    t_error_per_ast, t_error_per_token = result
    fprint(['validation set'])
    result = process_batches(validation_set, params, alpha, decay, True)
    if result is None:
        return
    v_error_per_ast, v_error_per_token = result

    dtpt = prev_t_token - t_error_per_token
    dtpa = prev_t_ast - t_error_per_ast
    dvpt = prev_v_token - v_error_per_token
    dvpa = prev_v_ast - v_error_per_ast
    print_str = [
        '################',
        'end of epoch {0} retry {1}'.format(epoch_num, retry_num),
        'train\t|\t{}\t|\t{}'.format(t_error_per_token, t_error_per_ast),
        'delta\t|\t{}\t|\t{}'.format(dtpt, dtpa),
        'validation\t|\t{}\t|\t{}'.format(v_error_per_token, v_error_per_ast),
        'delta\t|\t{}\t|\t{}'.format(dvpt, dvpa),
        '################',
        'new epoch'
    ]
    fprint(print_str, log_file)
    alpha *= 0.999

    new_params = open('new_params_t' + str(retry_num) + "_ep" + str(epoch_num), mode='wb')
    c_pickle.dump(params, new_params)
    return alpha, t_error_per_ast, t_error_per_token, v_error_per_ast, v_error_per_token


# @safe_run
def train_step(retry_num, batches, train_set_size, decay):
    def init_prev():
        return LEARN_RATE * (1 - MOMENTUM), 0, 0, 0, 0

    prev = init_prev()
    params = initialize()
    for train_epoch in range(20):
        prev = epoch_step(params, train_epoch, retry_num, prev, batches, train_set_size, decay)
        if prev is None:
            return


def main():
    dataset_dir = '../Dataset/'
    ast_file = open(dataset_dir + 'ast_file', mode='rb')
    data_ast = c_pickle.load(ast_file)
    batches = create_batches(data_ast[:2])
    decay = 5e-5
    train_set_size = 8  # (len(batches) // 10) * 8
    print(len(batches))
    for train_retry in range(20):
        train_step(train_retry, batches, train_set_size, decay)


if __name__ == '__main__':
    main()
    # build_asts('../Dataset/')
