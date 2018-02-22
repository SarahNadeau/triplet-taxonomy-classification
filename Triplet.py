import tensorflow as tf
import numpy as np
import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from tensorflow.python import debug as tf_debug


# load data from standardized train/test set
def load_data():
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_test150.pickle', 'rb') as f:
        x_test = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_train150.pickle', 'rb') as f:
        x_train = pickle.load(f)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_test150.pickle', 'rb') as f:
        y_test_str = pickle.load(f)
        y_test = enumerate_y_labels(y_test_str)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_train150.pickle', 'rb') as f:
        y_train_str = pickle.load(f)
        y_train = enumerate_y_labels(y_train_str)
        f.close()
    return x_test, x_train, y_test, y_train


# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(set(sorted(y_str)))])
    return [ylabel_dict[x] for x in y_str]


# binarize DNA sequences (takes in sequence)
def seq2binary(seq):
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
    return binary_kmer


# return lists of samples, positive examples, and negative examples of the specified batch size
def get_triplet_batch(sample_dict, batch_size):
    xs = []
    xps = []
    xns = []
    for i in range(0, batch_size):
        num_classes = len(sample_dict.keys())
        p_class, n_class = np.random.choice(range(0, num_classes), 2, replace=False)
        x_ind, p_ind = np.random.choice(range(0, len(sample_dict[p_class])), 2, replace=False)
        n_ind = np.random.choice(range(0, len(sample_dict[n_class])))
        xs.append(seq2binary(sample_dict[p_class][x_ind]))
        xps.append(seq2binary(sample_dict[p_class][p_ind]))
        xns.append(seq2binary(sample_dict[n_class][n_ind]))
    return xs, xps, xns


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
def compute_euclidean_distances(x, y, w=None):
    d = tf.square(tf.subtract(x, y))
    if w is not None:
        d = tf.transpose(tf.multiply(tf.transpose(d), w))
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d


class Triplet:
    def __init__(self, kmer_len, alpha):

        self.kmer_len = kmer_len
        self.alpha = alpha  # alpha is the margin for the loss function

        # make placeholders for all tensorflow variables
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='x')
            self.xp = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xp')
            self.xn = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xn')
            self.pos_weights = tf.placeholder_with_default(shape=[None], name='pos_weights', input=np.array([]))
            self.neg_weights = tf.placeholder_with_default(shape=[None], name='neg_weights', input=np.array([]))
            self.top_k = tf.placeholder_with_default(shape=[], name='top_k_losses', input=75)

        # embed samples
        with tf.variable_scope('embedding') as scope:
            self.o = self.__embedding_network(self.x, kmer_len)
            scope.reuse_variables()
            self.op = self.__embedding_network(self.xp, kmer_len)
            self.on = self.__embedding_network(self.xn, kmer_len)

        # compute distances between sample embeddings
        with tf.variable_scope('distances'):
            self.dp = compute_euclidean_distances(self.o, self.op)
            self.dn = compute_euclidean_distances(self.o, self.on)

        # define loss function
        with tf.variable_scope('loss'):
            self.loss = tf.nn.relu(tf.pow(self.dp, 2) - tf.pow(self.dn, 2) + alpha)
            self.loss = -tf.reduce_mean(tf.nn.top_k(-self.loss, k=self.top_k).values)
            # doesn't help to eliminate top_k

    # define convolutional network layers
    def __embedding_network(self, x, kmer_len):

        dim = seq_dim
        with tf.variable_scope('conv1'):
            out = 32
            w = weight_variable([3, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv2'):
            out = 64
            w = weight_variable([3, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('conv3'):
            out = 128
            w = weight_variable([3, dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(conv1d(x, w) + b)
            dim = out
            x = h

        with tf.variable_scope('readout'):
            gpool = tf.nn.pool(x, [h.get_shape()[1]], pooling_type="AVG", padding="VALID", name="gpool")
            return tf.reshape(gpool, [-1, embed_dim])


# SET PARAMETERS HERE
k_mer_len = 150
batch_size = 1
logging_frequency = 10
iterations = 10
margin = 1
top_k_iter_start = 300
n_easiest = batch_size
seq_dim = 4
test_num = 10
embed_dim = 128

# load data, print summary
x_test, x_train, y_test, y_train = load_data()
print("{} training samples".format(len(x_train)))
print("{} testing samples".format(len(x_test)))

# build training sample dictionary to group reads by organism
# keys are integer y_train labels, values are list of corresponding x_train samples
x_train_dict = {}
for i in range(0, len(x_train)):
    if y_train[i] not in x_train_dict:
        x_train_dict[y_train[i]] = [x_train[i]]
    else:
        x_train_dict[y_train[i]].append(x_train[i])

# create graph
triplet = Triplet(kmer_len=k_mer_len, alpha=margin)
train_step = tf.train.AdamOptimizer(10e-5).minimize(triplet.loss)

# train model
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    print("...training  model")
    for i in range(iterations):
        top_k = batch_size if i < top_k_iter_start else n_easiest
        batch = get_triplet_batch(x_train_dict, batch_size)
        if i % logging_frequency == 0:
            loss = sess.run(triplet.loss, feed_dict={triplet.top_k: n_easiest, triplet.x: batch[0],
                                                     triplet.xp: batch[1], triplet.xn: batch[2]})
            print('step %d, training loss %g' % (i, loss))
        train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1],
                                  triplet.xn: batch[2]})

    # embed training reads -- train_embeddings is a list of 128D sequence embeddings
    print("...classifying")
    train_reads = []
    for train_read in tqdm(x_train):
        train_reads.append(seq2binary(train_read))
        train_embeddings = sess.run(triplet.o, feed_dict={triplet.x: train_reads})

    # embed test reads -- test_embeddings is a list of 128D sequence embeddings
    test_reads = []
    for test_read in tqdm(x_test):
        test_reads.append(seq2binary(test_read))
        test_embeddings = sess.run(triplet.o, feed_dict={triplet.x: test_reads})

# find knn for test reads among train reads
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(train_embeddings, y_train)
print("KNN score: {}".format(knn_model.score(test_embeddings, y_test)))



