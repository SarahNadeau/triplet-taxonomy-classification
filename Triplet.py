# This script takes in  DNA sequences and performs vector embeddings with a CNN according to an l2
# triplet loss function.
    # Currently working on taking n-gram frequencies as input vector instead

import tensorflow as tf
import numpy as np
import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import random
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise
from matplotlib.lines import Line2D


# input parameters
DATA_LOC = '/Users/nadeau/Documents/MS_research/Metagenome_Classification/Classification_Test_Data/Triplet_data/'
SEQ_LEN = 150
NUM_TRAIN = 400
NUM_TRAIN_GENOMES = 3
NUM_TEST = 80
SEQ_DIM = 4  # 4 if raw sequence input, 1 if vectorized by n-gram frequency input

# network parameters
EMBED_DIM = 128
WINDOW_SIZE = 3
BATCH_SIZE = 32
PULL_TO_PUSH = 0.7 # between 0 and 1
MARGIN = 3
ITERATIONS = 200
LOG_FREQ = 10
TOP_K_ITER_START = 200
K_HARDEST = BATCH_SIZE

# classification parameters
N_NEIGHBORS = 10


# load data from standardized train/test set as 150bp DNA sequences {A, C, T, G}
def load_data():
    with open(DATA_LOC + '{}bp_{}seqs_TEST.pickle'.format(SEQ_LEN, NUM_TEST), 'rb') as f:
        test_dict = pickle.load(f)
        f.close()
    with open(DATA_LOC + '{}bp_{}seqs_TRAIN.pickle'.format(SEQ_LEN, NUM_TRAIN), 'rb') as f:
        train_dict = pickle.load(f)
        f.close()

    return train_dict, test_dict


# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(sorted(set(y_str)))])
    reverse_ylabel_dict = dict([(x, y) for x, y in enumerate(sorted(set(y_str)))])
    return [ylabel_dict[x] for x in y_str], reverse_ylabel_dict


# binarize DNA sequences (takes in sequence)
def seq2binary(seq):
    # note: currently train and test samples chosen from genomes so as to not contain "N" bases
    count_n = 0
    binary_kmer = np.zeros((len(seq), SEQ_DIM))
    for i in range(0, len(seq)):
        c = seq[i]
        if c == 'A':
            binary_kmer[i, 0] = 1
        elif c == 'C':
            binary_kmer[i, 1] = 1
        elif c == 'T':
            binary_kmer[i, 2] = 1
        elif c == 'G':
            binary_kmer[i, 3] = 1
        elif c == 'N':
            binary_kmer[i, 4] = 1
            count_n += 1
    if count_n > 0:
        print("{} occurrences of N in sequence".format(count_n))
    return binary_kmer


# return lists of samples, positive examples, and negative examples of the specified batch size
def get_triplet_batch(sample_dict, sample_label_dict, batch_size):
    xs = np.zeros(shape=(batch_size, SEQ_LEN, SEQ_DIM), dtype=np.float64)
    xps = np.zeros(shape=(batch_size, SEQ_LEN, SEQ_DIM), dtype=np.float64)
    xns = np.zeros(shape=(batch_size, SEQ_LEN, SEQ_DIM), dtype=np.float64)

    for i in range(0, batch_size):
        num_classes = len(sample_dict.keys())
        p_class, n_class = random.sample(range(0, num_classes), 2)

        x_ind, p_ind = random.sample(range(len(sample_dict[sample_label_dict[p_class]])), 2)
        n_ind = random.randint(0, len(sample_dict[sample_label_dict[n_class]]) - 1)

        # for raw sequence input - need sample_label_dict to translate between filname y labels and numerical y labels
        xs[i, :] = seq2binary(sample_dict[sample_label_dict[p_class]][x_ind])
        xps[i, :] = seq2binary(sample_dict[sample_label_dict[p_class]][p_ind])
        xns[i, :] = seq2binary(sample_dict[sample_label_dict[n_class]][n_ind])

        # for vectorized input
        # xs.append(sample_dict[p_class][x_ind])
        # xps.append(sample_dict[p_class][p_ind])
        # xns.append(sample_dict[n_class][n_ind])

    return np.asarray(xs), np.asarray(xps), np.asarray(xns)


# update weights for convolutional filter
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable("weights", dtype=tf.float32, initializer=initial)


# update bias for convolutional filter
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.get_variable("biases", dtype=tf.float32, initializer=initial)


# perform the convolution
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')


# compute distances between two sample embeddings in each dimension
def compute_distances(x, y, w=None):
    x = tf.check_numerics(x, "got passed NaN to compute euclidean distance")
    y = tf.check_numerics(y, "got passed NaN to compute euclidean distance")
    d = tf.subtract(x, y)
    if w is not None:
        d = tf.transpose(tf.multiply(tf.transpose(d), w))
    return d


class Triplet:
    def __init__(self, kmer_len):

        self.kmer_len = kmer_len

        # make placeholders for all tensorflow variables
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, kmer_len, SEQ_DIM], name='x')
            self.xp = tf.placeholder(tf.float32, shape=[None, kmer_len, SEQ_DIM], name='xp')
            self.xn = tf.placeholder(tf.float32, shape=[None, kmer_len, SEQ_DIM], name='xn')

            self.x = tf.check_numerics(self.x, "x is NaN")
            self.xp = tf.check_numerics(self.xp, "xp is NaN")
            self.xn = tf.check_numerics(self.xn, "xn is NaN")

            self.pos_weights = tf.placeholder_with_default(shape=[None], name='pos_weights', input=np.array([]))
            self.neg_weights = tf.placeholder_with_default(shape=[None], name='neg_weights', input=np.array([]))
            self.top_k = tf.placeholder_with_default(shape=[], name='top_k_losses', input=10)

        # embed samples
        with tf.variable_scope('embedding') as scope:
            self.o = self.__embedding_network(self.x, kmer_len)
            scope.reuse_variables()
            self.op = self.__embedding_network(self.xp, kmer_len)
            self.on = self.__embedding_network(self.xn, kmer_len)

        # compute distances between sample embeddings
        with tf.variable_scope('distances'):
            self.o = tf.check_numerics(self.o, "o is NaN")
            self.op = tf.check_numerics(self.op, "op is NaN")
            self.on = tf.check_numerics(self.on, "on is NaN")

            # find o - op and o - on
            self.dp = compute_distances(self.o, self.op)
            self.dn = compute_distances(self.o, self.on)

            # NOTE: using euclidean distance fails because derivative around zero is vertical, loss goes to NaN
            # instead using l2 norm
            # find l2 norm of o - op and o - on
            self.np = tf.norm(self.dp, axis=1)
            self.nn = tf.norm(self.dn, axis=1)

        # define loss function
        with tf.variable_scope('loss'):
            # self.losses = tf.nn.relu(self.dp - (self.dn - alpha)) # + 0.1*self.dp
            self.losses = tf.nn.relu((1 - PULL_TO_PUSH) * self.np - (PULL_TO_PUSH * self.nn - MARGIN))
            # self.loss = tf.reduce_mean(self.losses)
            # self.top_k_losses = tf.nn.top_k(self.losses, k=self.top_k).values
            self.loss = tf.reduce_mean(tf.nn.top_k(self.losses, k=self.top_k).values)

    # define convolutional network layers
    def __embedding_network(self, x, kmer_len):

        dim = SEQ_DIM
        with tf.variable_scope('conv1'):
            out = 32
            w = weight_variable([WINDOW_SIZE, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv2'):
            out = 64
            w = weight_variable([WINDOW_SIZE, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv3'):
            out = 128
            w = weight_variable([WINDOW_SIZE, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('readout'):
            gpool = tf.nn.pool(x, [h.get_shape()[1]], pooling_type="AVG", padding="VALID", name="gpool")
            return tf.reshape(gpool, [-1, EMBED_DIM])


def main():

    # load data, print summary
    train_dict, test_dict = load_data()

    # enumerate y labels
    train_labels = train_dict.keys()
    test_labels = test_dict.keys()

    enumerated_train_labels = np.arange(len(train_labels))
    enumerated_test_labels = np.arange(len(test_labels))

    train_label_dict = dict(zip(enumerated_train_labels, train_labels))
    test_label_dict = dict(zip(enumerated_test_labels, test_labels))

    # create graph
    triplet = Triplet(kmer_len=SEQ_LEN)
    train_step = tf.train.AdamOptimizer(10e-4).minimize(triplet.loss)

    # train model with top-k
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        print("...training  model")
        for i in range(ITERATIONS):
            top_k = BATCH_SIZE if i < TOP_K_ITER_START else K_HARDEST
            batch = get_triplet_batch(train_dict, train_label_dict, BATCH_SIZE)
            if i % LOG_FREQ == 0:
                loss = sess.run(triplet.loss, feed_dict={triplet.top_k: top_k, triplet.x: batch[0],
                                                              triplet.xp: batch[1], triplet.xn: batch[2]})
                print('step %d, training loss %g' % (i, loss))
            train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0],
                                      triplet.xp: batch[1], triplet.xn: batch[2]})

        # embed training reads -- train_embeddings is a list of 128D sequence embeddings
        print("...classifying")
        # build training sample embedding dictionary to group read embeddings by organism
        # keys are integer y_train labels, values are list of corresponding x_train samples
        x_train_reads = []
        y_train = []
        for y_int in train_dict.keys():
            for seq in train_dict[y_int]:
                x_train_reads.append(seq2binary(seq))
                y_train.append(y_int)

        x_train_embeddings = sess.run(triplet.o, feed_dict={triplet.x: x_train_reads})

        x_test_reads = []
        y_test = []
        for y_int in test_dict.keys():
            for seq in test_dict[y_int]:
                x_test_reads.append(seq2binary(seq))
                y_test.append(y_int)

        x_test_embeddings = sess.run(triplet.o, feed_dict={triplet.x: x_test_reads})

    # find knn for test reads among train reads
    knn_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    print(np.asarray(x_train_embeddings).shape)
    print(np.asarray(x_test_embeddings).shape)
    # print(np.reshape(np.asarray(x_train_embeddings), (1200, 128)).shape)
    knn_model.fit(x_train_embeddings, y_train)
    print("Testing Error: {}".format((1 - knn_model.score(x_test_embeddings, y_test))*100))
    ### currently 100% because test sequence has diff label than train sequences

    # calculate cluster distances and standard deviations
    # index = []
    # for y_id in train_dict.keys():
    #     index.append(y_id)
    # means_df = pd.DataFrame(index=train_dict.keys(), columns=np.arange(EMBED_DIM))
    # for y_id in train_dict.keys():
    #     mean = np.mean(x_train_embedding_dict[y_id], axis=0)
    #     means_df.loc[y_id] = mean
    ### currently broken because no embedding dictionary

    # distances = pairwise.euclidean_distances(means_df)
    # print("pairwise distances between centroids of clusters:")
    # print(pd.DataFrame(distances, index=index, columns=index))
    ### currently broken because no embedding dictionary

    # make PCA of samples before embedding for comparison
    fig, ax = plt.subplots()
    x_train_reads = np.asarray(x_train_reads)
    x_train_reads = np.reshape(x_train_reads, (x_train_reads.shape[0], x_train_reads.shape[1]*x_train_reads.shape[2]))
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pca.fit_transform(x_train_reads)

    cmap = plt.get_cmap('viridis')
    unique_ys = pd.Series(list(y_train)).unique()
    colors =cmap(np.linspace(0, 1, len(unique_ys)))
    color_dict = dict(zip(unique_ys, colors))

    plot_colors = []
    for y in list(y_train):
        plot_colors.append(color_dict[y])

    legend_elements = []
    for i in range(0, len(colors)):
        legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label=unique_ys[i], markerfacecolor=tuple(colors[i]), markersize=5))

    ax.scatter(transformed[:, 0], transformed[:, 1], c=plot_colors)
    ax.legend(handles=legend_elements)
    ax.set_title("PCA of Binary Training Sequences")
    plt.show(fig)


    # plot separation of embeddings from training genomes
    fig2, ax = plt.subplots()
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pca.fit_transform(x_train_embeddings)

    ax.scatter(transformed[:, 0], transformed[:, 1], c=plot_colors)
    ax.legend(handles=legend_elements)
    ax.set_title("PCA of Training Sequence 128-D Embeddings")
    plt.show(fig2)

    # plot separation of embeddings from testing genomes
    fig3, ax = plt.subplots()
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pca.fit_transform(x_test_embeddings)

    unique_ys = pd.Series(list(y_test)).unique()
    colors = cmap(np.linspace(0, 1, len(unique_ys)))
    color_dict = dict(zip(unique_ys, colors))

    plot_colors = []
    for y in list(y_test):
        plot_colors.append(color_dict[y])

    legend_elements = []
    for i in range(0, len(colors)):
        legend_elements.append(
            Line2D([0], [0], marker='o', color=colors[i], label=unique_ys[i], markerfacecolor=tuple(colors[i]),
                   markersize=5))

    ax.scatter(transformed[:, 0], transformed[:, 1], c=plot_colors)
    ax.legend(handles=legend_elements)
    ax.set_title("PCA of Testing Sequence 128-D Embeddings")
    plt.show(fig3)


if __name__ == "__main__":
    main()



