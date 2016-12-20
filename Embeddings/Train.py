import _pickle as c_pickle
import os
from copy import deepcopy
from random import shuffle
import theano
from theano import shared
import numpy as np
from AST.Sampler import PreparedAST, generate_samples, build_asts
from Embeddings.Construct import construct
from Embeddings.Evaluation import EvaluationSet, process_network, EvaluationSet
from Embeddings.InitEmbeddings import initialize
from Embeddings.Parameters import *
from Utils.Visualization import new_figure, update_figure, save_to_file
from Utils.Wrappers import *
from collections import namedtuple
import gc

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'

# 0 because of preparing
training_token_index = 0
log_file = open('log.txt', mode='w')


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


# @timing
def prepare_net(eval_set: EvaluationSet, params, is_validation, index):
    if not is_validation:
        ast = eval_set.sample
        if eval_set.back_prop is None:
            fprint(['build ', index])
            eval_set.back_prop = construct(ast.positive, params, ast.training_token_index, is_validation)
    else:
        ast = eval_set.sample
        if eval_set.validation is None:
            fprint(['build ', index])
            eval_set.validation = construct(ast.positive, params, ast.training_token_index, is_validation)
    return eval_set


# @timing
def process_batch(batch, params, alpha, decay, is_validation, index, zero_emb):
    total_t_err = 0
    total_a_err = 0
    # samples_size = 1
    samples_size = len(batch)
    for ind, sample in enumerate(batch):
        # print(ind)
        eval_set = prepare_net(sample, params, is_validation, index)
        ept, err = process_network(eval_set, params, alpha, decay, is_validation, zero_emb)
        total_t_err += ept
        total_a_err += err

    # eval_set = prepare_net(batch[1], params, is_validation)
    # total_t_err, total_a_err = process_network(eval_set, params, alpha, decay, is_validation)

    return total_a_err / samples_size, total_t_err / samples_size


@safe_run
def process_batches(batches, params, alpha, decay, is_validation, zero_emb):
    error_per_ast = 0
    error_per_token = 0
    for i, batch in enumerate(batches):
        epa, ept = process_batch(batch, params, alpha, decay, is_validation, i, zero_emb)
        # message = ['\t\t|\t{}\t|\t{}\t|\t{}'.format(ept, epa, i)]
        # fprint(message)
        error_per_ast += epa
        error_per_token += ept
    return error_per_ast, error_per_token


def create_batches(data):
    samples = []
    for ast in data:
        generate_samples(ast, samples, training_token_index)
    data_size = len(samples)
    for i in range(data_size):
        samples[i] = EvaluationSet(samples[i])
    data_size //= BATCH_SIZE
    batches = [samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(data_size)]
    return batches


@safe_run
def epoch_step(params, epoch_num, retry_num, tparams, batches, train_set_size, decay, start_emb, zero_emb):
    shuffle(batches)
    train_set = batches[:train_set_size]
    # validation_set = batches[train_set_size + 1:]
    alpha, prev_t_ast, prev_t_token, prev_v_ast, prev_v_token, _ = tparams
    v_error_per_ast = 0
    v_error_per_token = 0
    fprint(['train set'])
    result = process_batches(train_set, params, alpha, decay, False, zero_emb)
    if result is None:
        return
    t_error_per_ast, t_error_per_token = result
    # fprint(['validation set'])
    # result = process_batches(validation_set, params, alpha, decay, True)
    # if result is None:
    #     return
    # v_error_per_ast, v_error_per_token = result

    dtpt = prev_t_token - t_error_per_token
    dtpa = prev_t_ast - t_error_per_ast
    # dvpt = prev_v_token - v_error_per_token
    # dvpa = prev_v_ast - v_error_per_ast
    print_str = [
        'end of epoch {0} retry {1}'.format(epoch_num, retry_num),
        'train\t|\t{}\t|\t{}'.format(t_error_per_token, t_error_per_ast),
        'delta\t|\t{}\t|\t{}'.format(dtpt, dtpa),
        # 'validation\t|\t{}\t|\t{}'.format(v_error_per_token, v_error_per_ast),
        # 'delta\t|\t{}\t|\t{}'.format(dvpt, dvpa),
        '################'
    ]
    fprint(print_str, log_file)
    for name, emb in params.embeddings.items():
        if np.array_equal(emb.eval(), start_emb[name].eval()):
            fprint(['embedding is equal to start: ', name], log_file)
            params.embeddings[name] = zero_emb

    alpha *= 0.999
    if epoch_num % 100 == 0:
        with open('Embeddings/NewParams/new_params_t' + str(retry_num) + "_ep" + str(epoch_num),
                  mode='wb') as new_params:
            c_pickle.dump(params, new_params)

    train_size = len(train_set)
    return TrainingParams(alpha, t_error_per_ast, t_error_per_token, v_error_per_ast, v_error_per_token,
                          t_error_per_ast / train_size)


TrainingParams = namedtuple('TrainingParams',
                            ['alpha', 'prev_t_ast', 'prev_t_token', 'prev_v_ast', 'prev_v_token', 'error_to_print'])


def reset_batches(batches):
    for batch in batches:
        for block in batch:
            block.back_prop = None
            block.validation = None
    gc.collect()


@safe_run
def train_step(retry_num, batches, train_set_size, token_set, decay):
    tparams = TrainingParams(LEARN_RATE * (1 - MOMENTUM), 0, 0, 0, 0, 0)
    nparams = initialize(token_set)
    zero_emb = shared(np.zeros(NUM_FEATURES).astype(theano.config.floatX))
    start_emb = deepcopy(nparams.embeddings)
    reset_batches(batches)
    plot_axes, plot = new_figure(retry_num, EPOCH_IN_RETRY, 1.1)

    for train_epoch in range(EPOCH_IN_RETRY):
        tparams = epoch_step(nparams, train_epoch, retry_num, tparams, batches, train_set_size, decay, start_emb,
                             zero_emb)
        if tparams is None:
            break
        update_figure(plot, plot_axes, train_epoch, tparams.error_to_print, -1)

    save_to_file(plot, 'retry{}.png'.format(retry_num))


@sig_handler
def main():
    dataset_dir = 'Dataset/'
    with open(dataset_dir + 'ast_file', mode='rb') as ast_file:
        data_ast = c_pickle.load(ast_file)
    token_set = data_ast.token_set
    batches = create_batches(data_ast.ast_set)
    batches = batches[0:SAMPLES_AMOUNT // BATCH_SIZE]
    decay = 5e-5
    # train_set_size = (len(batches) // 10) * 8
    train_set_size = len(batches) - 1
    print(len(batches))
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, batches, train_set_size, token_set, decay)


if __name__ == '__main__':
    gc.enable()
    main()
    # build_asts('../Dataset/java_files/')
