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
    model_name = 'TFAuthorClassifier/TestData/model-43'
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open('Dataset/CombinedProjects/top_authors_MPS_data', 'rb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'rb') as f:
        data_set = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(data_set.amount)
    updates, net, summaries = build_net(params)
    test_set = generate_batches(data_set.test, emb_indexes, data_set.r_index, net)
    cm = ConfMatrix(data_set.amount)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        saver.restore(sess, model_name)
        for feed in test_set:
            res, tar = sess.run(fetches=[net.result, net.target], feed_dict=feed)
            for res in zip(res, tar):
                cm.add(*res)
    cm.calc()
    print(str(cm))
    cm.print_conf_matrix()


if __name__ == '__main__':
    main()
