import sys
import _pickle as P

import logging
import tensorflow as tf
from TFAuthorClassifier.BatchBuilder import generate_batches
from TFAuthorClassifier.DataPreparation import DataSet
from TFAuthorClassifier.NetBuilder import build_net
from TFAuthorClassifier.TFParameters import init_params
from Utils.ConfMatrix import ConfMatrix


def main():
    model_name = 'TFAuthorClassifier/NewParams/model'
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open('Dataset/CombinedProjects/top_authors_MPS_data', 'rb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'rb') as f:
        data_set = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(data_set.amount)
    updates, net, summaries = build_net(params)
    test_set = generate_batches(data_set.test, emb_indexes, data_set.r_index, net)
    targets = [pc.target for pc in test_set[0].keys()]
    cm = ConfMatrix(data_set.amount)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        saver.restore(sess, model_name)
        for feed in test_set:
            res = sess.run(fetches=tf.arg_max(net.out,1), feed_dict=feed)
            tar = sess.run(fetches=targets, feed_dict=feed)
            for i, r in enumerate(res):
                cm.add(r, tar[i])
    cm.calc()
    print(str(cm))
    cm.print_conf_matrix()


if __name__ == '__main__':
    main()
