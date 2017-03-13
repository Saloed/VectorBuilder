import _pickle as P
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


def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open('Dataset/CombinedProjects/top_authors_MPS_data', 'rb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'rb') as f:
        data_set = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(data_set.amount)
    updates, net, summaries = build_net(params)
    train_set = generate_batches(data_set.train, emb_indexes, data_set.r_index, net)
    test_set = generate_batches(data_set.valid, emb_indexes, data_set.r_index, net)
    saver = tf.train.Saver()
    for retry_num in range(NUM_RETRY):
        save_path = 'TFAuthorClassifier/Params/Retry_{}/'.format(retry_num)
        plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 2)
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter('TFAuthorClassifier/Summary', sess.graph)
            sess.run(tf.global_variables_initializer())
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
                logging.info('\n'.join(print_str))
                # if train_epoch % SAVE_PERIOD == 0:
                if True:
                    saver.save(sess, save_path + 'model', train_epoch)
                update_figure(plot, plot_axes, train_epoch, te_loss, tr_loss)
                info = sess.run(fetches=summaries)
                summary_writer.add_summary(info, train_epoch)
        save_to_file(plot, save_path + 'error.png'.format(retry_num))
        summary_writer.close()


if __name__ == '__main__':
    main()
