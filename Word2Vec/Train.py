import os
import numpy as np
import tensorflow as tf
from tensorflow.models.embedding import word2vec_optimized as word2vec
from tensorflow.models.embedding.word2vec_optimized import Word2Vec
import _pickle as P
from AST.Sampler import build_psi_text
from AuthorClassifier.ClassifierParams import NUM_FEATURES
from Embeddings.Parameters import Parameters
from theano import shared


def make_train_file(dataset_dir):
    build_psi_text(dataset_dir)


def convert_to_parameters(emb, storage):
    embeddings = {cut_word(word): shared(emb[i], cut_word(word)) for word, i in storage._word2id.items()}
    # print(embeddings)
    param = Parameters(None, None, None, embeddings)
    with open('embeddings_w2v_new', 'wb') as f:
        P.dump(param, f)


def cut_word(word):
    return str(word)[2:-1]


def parse_model():
    with open('SaveDir/end_w2v_model', 'rb') as f:
        storage = P.load(f)  # type: W2VStorage

    emb_dim = storage.options.emb_dim
    vocab_size = storage.options.vocab_size

    w_in = tf.Variable(tf.zeros([vocab_size, emb_dim]), name="w_in")
    w_out = tf.Variable(tf.zeros([vocab_size, emb_dim]), name="w_out")
    global_step = tf.Variable(0, name="global_step")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "SaveDir/end_model.ckpt-165217957")
        emb = w_in.eval(sess)  # type: np.multiarray.ndarray
        convert_to_parameters(emb, storage)


class W2VStorage:
    def __init__(self, options, w2i, i2w):
        self.options = options
        self._word2id = w2i
        self._id2word = i2w


def dump_model(model: Word2Vec, name):
    with open('SaveDir/' + name, 'wb') as f:
        P.dump(W2VStorage(model._options, model._word2id, model._id2word), f)


def main():
    word2vec.FLAGS.save_path = 'SaveDir'
    word2vec.FLAGS.checkpoint_interval = 100
    word2vec.FLAGS.epochs_to_train = 500
    word2vec.FLAGS.embedding_size = NUM_FEATURES
    # word2vec.FLAGS.concurrent_steps = 4
    word2vec.FLAGS.train_data = '../Dataset/psi_text.data'
    op = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(op, session)
            dump_model(model, 'start_w2v_model')

            for _ in range(op.epochs_to_train):
                model.train()  # Process one epoch

            model.saver.save(session,
                             os.path.join(op.save_path, "end_model.ckpt"),
                             global_step=model.global_step)
            dump_model(model, 'end_w2v_model')


if __name__ == '__main__':
    # main()
    parse_model()
    # make_train_file('../Dataset/CombinedProjects/MPS')
