import _pickle as P
import datetime
import os
import logging
import sys
from random import shuffle
from TFAuthorClassifier.BatchBuilder import generate_batches
from TFAuthorClassifier.DataPreparation import DataSet
from TFAuthorClassifier.NetBuilder import build_net
from TFAuthorClassifier.TFParameters import *
from Utils.Visualization import new_figure, update_figure, save_to_file
from Utils.Wrappers import timing


@timing
def process_set(batches, fun, is_train, session):
    res = ([], [], [])
    for feed in batches:
        if is_train:
            loss, max_loss, err, _ = session.run(fetches=fun, feed_dict=feed)
        else:
            loss, max_loss, err = session.run(fetches=fun, feed_dict=feed)
        res[0].append(float(loss))
        res[1].append(float(max_loss))
        res[2].append(float(err))
    loss = np.mean(res[0])
    loss_max = np.max(res[1])
    err = np.mean(res[2])
    return loss, loss_max, err


def write_parameters(path):
    from TFAuthorClassifier import TFParameters
    params = dir(TFParameters)
    param_names = [v for v in params if v.isupper() and len(v) > 1]
    param_values = [getattr(TFParameters, v) for v in param_names]
    param_str = '\n'.join(('{0} = {1}'.format(p[0], p[1]) for p in zip(param_names, param_values)))
    with open(path + 'parameters.txt', 'w') as f:
        f.write(param_str)


def main():
    with open('Dataset/platform_data_set_100_100', 'rb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'rb') as f:
        data_set = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(data_set.amount)
    updates, net, summaries = build_net(params)

    train_set = generate_batches(data_set.train, emb_indexes, data_set.r_index, net, DROPOUT)
    test_set = generate_batches(data_set.valid, emb_indexes, data_set.r_index, net, 1.0)
    current_date = datetime.datetime.now()
    current_date = '{}_{}_{}'.format(current_date.day, current_date.month, current_date.year)
    base_path = 'Results/platform_parallel_400_{}/'.format(current_date)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    write_parameters(base_path)
    for retry_num in range(NUM_RETRY):
        save_path = base_path + 'retry_{}/'.format(retry_num)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger = logging.getLogger('NetLogger_{}'.format(retry_num))
        logger.addHandler(logging.FileHandler(base_path + 'log_{}.txt'.format(retry_num)))
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)
        saver = tf.train.Saver(max_to_keep=NUM_EPOCH)
        plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 2)
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter(save_path + 'Summary', sess.graph)
            sess.run(tf.global_variables_initializer())
            try:
                for train_epoch in range(NUM_EPOCH):
                    shuffle(train_set)
                    tr_loss, tr_max, tr_err = process_set(train_set, [net.loss, net.max_loss, net.error, updates], True,
                                                          sess)
                    te_loss, te_max, te_err = process_set(test_set, [net.loss, net.max_loss, net.error], False, sess)
                    print_str = [
                        'epoch {0} retry {1}'.format(train_epoch, retry_num),
                        'train | mean {0:.4f} | max {1:.4f} | percent {2:.2f}'.format(float(tr_loss),
                                                                                      float(tr_max),
                                                                                      float(tr_err)),
                        'test  | mean {0:.4f} | max {1:.4f} | percent {2:.2f}'.format(float(te_loss),
                                                                                      float(te_max),
                                                                                      float(te_err)),
                        '################'
                    ]
                    logger.info('\n'.join(print_str))
                    # if train_epoch % SAVE_PERIOD == 0:
                    if True:
                        saver.save(sess, save_path + 'model', train_epoch)
                    update_figure(plot, plot_axes, train_epoch, te_loss, tr_loss)
                    info = sess.run(fetches=summaries)
                    summary_writer.add_summary(info, train_epoch)
            except Exception as ex:
                logger.error(str(ex))
                logger.error(str(ex.__traceback__))
            finally:
                save_to_file(plot, base_path + 'error_{}.png'.format(retry_num))
                summary_writer.close()


if __name__ == '__main__':
    main()
