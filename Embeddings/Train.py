import _pickle as c_pickle
from random import shuffle
import matplotlib.pyplot as plotter
import theano
import numpy as np
from AST.Sampler import PreparedAST, generate_samples
from Embeddings.Construct import construct
from Embeddings.Evaluation import EvaluationSet, process_network, EvaluationSet
from Embeddings.InitEmbeddings import initialize
from Embeddings.Parameters import *
from Utils.Wrappers import *
from collections import namedtuple

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'

# 0 because of preparing
training_token_index = 0
log_file = open('log.txt', mode='w')
batch_size = 10


def fprint(print_str: list, file=log_file):
    return
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


# @timing
def prepare_net(eval_set: EvaluationSet, params, is_validation):
    if not is_validation:
        ast = eval_set.sample
        if eval_set.back_prop is None:
            eval_set.back_prop = construct(ast.positive, params, ast.training_token_index, is_validation)
    else:
        ast = eval_set.sample
        if eval_set.validation is None:
            eval_set.validation = construct(ast.positive, params, ast.training_token_index, is_validation)
    return eval_set


# @timing
def process_batch(batch, params, alpha, decay, is_validation):
    total_t_err = 0
    total_a_err = 0
    samples_size = 1
    # samples_size = len(batch)
    # for ind, sample in enumerate(batch):
    #     print(ind)
    #     eval_set = prepare_net(sample, params, is_validation)
    #     ept, err = process_network(eval_set, params, alpha, decay)
    #     total_t_err += ept
    #     total_a_err += err

    eval_set = prepare_net(batch[1], params, is_validation)
    total_t_err, total_a_err = process_network(eval_set, alpha, decay, is_validation)

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
    for i in range(data_size):
        samples[i] = EvaluationSet(samples[i])
    data_size //= batch_size
    batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(data_size)]
    return batches


# @safe_run
def epoch_step(params, epoch_num, retry_num, tparams, batches, train_set_size, decay):
    # shuffle(batches)
    train_set = batches[0:1]
    validation_set = batches[0:1]
    alpha, prev_t_ast, prev_t_token, prev_v_ast, prev_v_token = tparams
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

    new_params = open('NewParams/new_params_t' + str(retry_num) + "_ep" + str(epoch_num), mode='wb')
    c_pickle.dump(params, new_params)

    return TrainingParams(alpha, t_error_per_ast, t_error_per_token, v_error_per_ast, v_error_per_token)


TrainingParams = namedtuple('TrainingParams', ['alpha', 'prev_t_ast', 'prev_t_token', 'prev_v_ast', 'prev_v_token'])


def new_figure(num):
    x = np.arange(0, EPOCH_IN_RETRY, 1)
    y = np.full(EPOCH_IN_RETRY, -1.1)
    fig = plotter.figure(num)
    fig.set_size_inches(1920, 1080)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, EPOCH_IN_RETRY)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation error')
    ax.grid(True)
    line, = ax.plot(x, y, '|')
    fig.show(False)
    fig.canvas.draw()
    return line, fig


def update_figure(plot, axes, x, y):
    new_data = axes.get_ydata()
    new_data[x] = y
    axes.set_ydata(new_data)
    plot.canvas.draw()


def reset_batches(batches):
    for batch in batches:
        for block in batch:
            block.back_prop = None
            block.validation = None


# @safe_run
def train_step(retry_num, batches, train_set_size, decay):
    tparams = TrainingParams(LEARN_RATE * (1 - MOMENTUM), 0, 0, 0, 0)
    nparams = initialize()
    reset_batches(batches)
    plot_axes, plot = new_figure(retry_num)

    for train_epoch in range(EPOCH_IN_RETRY):
        tparams = epoch_step(nparams, train_epoch, retry_num, tparams, batches, train_set_size, decay)
        if tparams is None:
            return
        update_figure(plot, plot_axes, train_epoch, tparams.prev_v_ast)

    plot.savefig('NewParams/retry{}.png'.format(retry_num))
    plotter.close(plot)


def main():
    dataset_dir = '../Dataset/'
    with open(dataset_dir + 'ast_file', mode='rb') as ast_file:
        data_ast = c_pickle.load(ast_file)
    batches = create_batches(data_ast[0:100])
    decay = 5e-5
    train_set_size = 8  # (len(batches) // 10) * 8
    print(len(batches))
    for train_retry in range(20):
        train_step(train_retry, batches, train_set_size, decay)


if __name__ == '__main__':
    main()
    # build_asts('../Dataset/')
