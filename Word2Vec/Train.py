import _pickle as P
import numpy as np
import tensorflow as tf
from AST.GitAuthor import build_psi_text
from TFAuthorClassifier.TFParameters import NUM_FEATURES
from Word2Vec import word2vec_optimized as word2vec
from Word2Vec.word2vec_optimized import Word2Vec


def make_train_file(data_set_dir):
    build_psi_text(data_set_dir)


def main():
    word2vec.FLAGS.save_path = 'SaveDir'
    word2vec.FLAGS.train_data = '../Dataset/psi_text.data'
    word2vec.FLAGS.epochs_to_train = 500
    word2vec.FLAGS.embedding_size = NUM_FEATURES
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, "end_model.ckpt")
        emb = model.w_in.eval(session)  # type: np.multiarray.ndarray
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
        with open('embeddings_w2v_new', 'wb') as f:
            P.dump(embeddings, f)


if __name__ == '__main__':
    main()
    # make_train_file('../Dataset/CombinedProjects/MPS')
