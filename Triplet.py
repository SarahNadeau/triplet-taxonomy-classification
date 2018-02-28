# This script takes in short, raw DNA sequences and performs vector embeddings with a CNN according to an l2
# triplet loss function.
    # Currently working on taking n-gram frequencies as input vector instead

import tensorflow as tf
import numpy as np
import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from tensorflow.python import debug as tf_debug
import random
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt


# load data from standardized train/test set as raw sequences
def load_data():
    with open('/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/raw_train_test/X_test150.pickle', 'rb') as f:
        X_test = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/raw_train_test/X_train150.pickle', 'rb') as f:
        X_train = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/raw_train_test/y_test150.pickle', 'rb') as f:
        y_test_str = pickle.load(f)
        y_test = enumerate_y_labels(y_test_str)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/Classification_Test_Data/raw_train_test/y_train150.pickle', 'rb') as f:
        y_train_str = pickle.load(f)
        y_train = enumerate_y_labels(y_train_str)
        f.close()

    # to fetch standardized 150 bp train and test vectorized seqs:
    # X_test = np.genfromtxt(
    #     '/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/xTe.csv',
    #     delimiter=',')
    # X_train = np.genfromtxt(
    #     '/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/xTr.csv',
    #     delimiter=',')
    # y_test = np.genfromtxt(
    #     '/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/yTe.csv',
    #     delimiter=',')
    # y_train = np.genfromtxt(
    #     '/Users/nadeau/Documents/Metagenome_Classification/mlcircus-lmnn-5b49cafaeb9a/lmnn3/demos/yTr.csv',
    #     delimiter=',')

    return X_test, X_train, y_test, y_train

# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(sorted(set(y_str)))])
    return [ylabel_dict[x] for x in y_str]


# binarize DNA sequences (takes in sequence)
def seq2binary(seq):
    # note: currently train and test samples chosen from genomes so as to not contain "N" bases
    # count_n = 0
    binary_kmer = np.zeros((len(seq), seq_dim))
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
        # elif c == 'N':
        #     binary_kmer[i, 4] = 1
        #     count_n += 1
    # if count_n > 0:
    #     print("{} occurrences of N in sequence".format(count_n))
    return binary_kmer


# return lists of samples, positive examples, and negative examples of the specified batch size
def get_triplet_batch(sample_dict, batch_size):
    xs = np.zeros(shape=(batch_size, k_mer_len, seq_dim), dtype=np.float64)
    xps = np.zeros(shape=(batch_size, k_mer_len, seq_dim), dtype=np.float64)
    xns = np.zeros(shape=(batch_size, k_mer_len, seq_dim), dtype=np.float64)

    for i in range(0, batch_size):
        num_classes = len(sample_dict.keys())
        p_class, n_class = random.sample(range(0, num_classes), 2)

        x_ind, p_ind = random.sample(range(len(sample_dict[p_class])), 2)
        n_ind = random.randint(0, len(sample_dict[n_class]) - 1)

        # for raw sequence input
        xs[i, :] = seq2binary(sample_dict[p_class][x_ind])
        xps[i, :] = seq2binary(sample_dict[p_class][p_ind])
        xns[i, :] = seq2binary(sample_dict[n_class][n_ind])

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


# compute euclidean distances between two sample embeddings
# NOTE: using euclidean distance fails because derivative around zero is vertical, loss goes to NaN
# instead using l2 norm
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
            self.x = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='x')
            self.xp = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xp')
            self.xn = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xn')

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

            # find l2 norm of o - op and o - on
            self.np = tf.norm(self.dp, axis=1)
            self.nn = tf.norm(self.dn, axis=1)
            # self.np = tf.nn.l2_loss(self.dp)
            # self.nn = tf.nn.l2_loss(self.dn)

        # define loss function
        with tf.variable_scope('loss'):
            # self.losses = tf.nn.relu(self.dp - (self.dn - alpha)) # + 0.1*self.dp
            self.losses = tf.nn.relu((1 - pull_to_push) * self.np - (pull_to_push*self.nn - margin))
            # self.loss = tf.reduce_mean(self.losses)
            self.top_k_losses = tf.nn.top_k(self.losses, k=self.top_k).values
            self.loss = tf.reduce_mean(tf.nn.top_k(self.losses, k=self.top_k).values)

    # define convolutional network layers
    def __embedding_network(self, x, kmer_len):

        dim = seq_dim
        with tf.variable_scope('conv1'):
            out = 32
            w = weight_variable([window_size, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv2'):
            out = 64
            w = weight_variable([window_size, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv3'):
            out = 128
            w = weight_variable([window_size, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('readout'):
            gpool = tf.nn.pool(x, [h.get_shape()[1]], pooling_type="AVG", padding="VALID", name="gpool")
            return tf.reshape(gpool, [-1, embed_dim])


# SET PARAMETERS HERE
k_mer_len = 150
seq_dim = 4  # 5 if raw sequence input, 1 if vectorized by n-gram frequency input
embed_dim = 128
window_size = 10

batch_size = 32
pull_to_push = 0.5 # between 0 and 1
margin = 3
iterations = 200

logging_frequency = 10
top_k_iter_start = 200
n_hardest = batch_size

n_neighbors = 10


# load data, print summary
X_test, X_train, y_test, y_train = load_data()
print("{} training samples".format(len(X_train)))
print("{} testing samples".format(len(X_test)))

# build training sample dictionary to group reads by organism
# keys are integer y_train labels, values are list of corresponding x_train samples
x_train_dict = {}
for i in range(0, len(X_train)):
    if y_train[i] not in x_train_dict:
        x_train_dict[y_train[i]] = [X_train[i]]
    else:
        x_train_dict[y_train[i]].append(X_train[i])

# create graph
triplet = Triplet(kmer_len=k_mer_len)
train_step = tf.train.AdamOptimizer(10e-4).minimize(triplet.loss)

# train model with top-k
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    print("...training  model")
    for i in range(iterations):
        top_k = batch_size if i < top_k_iter_start else n_hardest
        batch = get_triplet_batch(x_train_dict, batch_size)
        if i % logging_frequency == 0:
            loss, losses, dp, dn, x, o, top_k_losses, normp, xp, xn = sess.run([triplet.loss, triplet.losses, triplet.dp, triplet.dn, triplet.x, triplet.o, triplet.top_k_losses, triplet.np, triplet.xp, triplet.xn],
                       feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1],
                                  triplet.xn: batch[2]})
            print('step %d, training loss %g' % (i, loss))
        train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1],
                                  triplet.xn: batch[2]})

    # embed training reads -- train_embeddings is a list of 128D sequence embeddings
    print("...classifying")
    train_reads = []
    for train_read in tqdm(X_train):
        train_reads.append(seq2binary(train_read))
    train_embeddings = sess.run(triplet.o, feed_dict={triplet.x: train_reads})

    # embed test reads -- test_embeddings is a list of 128D sequence embeddings
    test_reads = []
    for test_read in tqdm(X_test):
        test_reads.append(seq2binary(test_read))
    test_embeddings = sess.run(triplet.o, feed_dict={triplet.x: test_reads})

# find knn for test reads among train reads
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_model.fit(train_embeddings, y_train)
print("Testing Error: {}".format((1 - knn_model.score(test_embeddings, y_test))*100))

# plot separation of embeddings from genomes
pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
transformed = pca.fit_transform(train_embeddings)
y_colors = list(y_train)
plt.scatter(transformed[:, 0], transformed[:, 1], c=y_colors)
plt.title("PCA of train embeddings")
plt.show()



