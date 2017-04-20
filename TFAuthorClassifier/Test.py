import _pickle as P
import os

import tensorflow as tf

from TFAuthorClassifier.BatchBuilder import generate_batches
from TFAuthorClassifier.DataPreparation import DataSet
from TFAuthorClassifier.NetBuilder import build_net
from TFAuthorClassifier.TFParameters import init_params
from Utils.ConfMatrix import ConfMatrix


def main():
    model_name = 'Results/camel_parallel_400_12_4_2017/retry_0/model-21'
    test_name = 'Results/camel_test/'
    if not os.path.exists(test_name):
        os.makedirs(test_name)
    with open('Dataset/TestRepos/camel_data_set', 'rb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'rb') as f:
        data_set = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(data_set.amount)
    updates, net, summaries = build_net(params)
    test_set = generate_batches(data_set.test, emb_indexes, data_set.r_index, net, 1.0)
    cm = ConfMatrix(data_set.amount, test_name)
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
    with open(test_name + 'test.txt', 'w') as res_file:
        res_file.write(str(cm))
    cm.print_conf_matrix()


if __name__ == '__main__':
    main()
