import tensorflow as tf
import numpy as np
import os
from Bio import SeqIO
from random import shuffle, sample
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import _pickle as pickle
from math import sqrt
import sys

# def seq2binary(seq):
#     # N's = G's
#     binary_kmer = np.zeros((len(seq), 2))
#     for i in range(0, len(seq)):
#         c = seq[i]
#         if c == 'A' or c == 'C':
#             binary_kmer[i, 0] = 0
#         else:
#             binary_kmer[i, 0] = 1
#
#         if c == 'A' or c == 'T':
#             binary_kmer[i, 1] = 0
#         else:
#             binary_kmer[i, 1] = 1
#     return binary_kmer

def seq2binary(seq):
    binary_kmer = np.zeros((len(seq), 4))
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


class SeqDatasetGenerator(object):

    def __init__(self, kmers_by_genome_train, kmers_by_genome_test):
        self.kmers_by_genome_train = kmers_by_genome_train
        self.kmers_by_genome_test = kmers_by_genome_test

    @staticmethod
    def get_file_list():
        file_list = []  # GENOME FILES ADDED HERE

        for file_name in os.listdir('Genomes'):
            # print(file_name)
            file_name = os.path.join("Genomes", file_name)
            if file_name.endswith(".gbff") and not file_name.endswith("test.gbff"):
                file_list.append(file_name)
        return file_list

    @staticmethod
    def default_load(k):
        kmers_by_genome_train = []
        kmers_by_genome_test = []
        num_kmers = []

        file_list = SeqDatasetGenerator.get_file_list()

        for file in file_list:
            words = []
            for accession in SeqIO.parse(file, 'genbank'):
                print("filename {} has taxonomy {}".format(file, accession.annotations["taxonomy"]))
                # print(accession.id)
                seq = str(accession.seq)
                for s in range(0, len(seq) - k + 1):
                    words.append(seq[s:s + k])
            kmers_by_genome_train.append(words)

        # print("Number of genomes: {}".format(len(kmers_by_genome_train)))
        for list in kmers_by_genome_train:
            shuffle(list)
        for i in range(0, len(kmers_by_genome_train)):
            num_kmers.append(len(kmers_by_genome_train[i]))
            print("# k-mers in genome {0}: {1}".format(i, len(kmers_by_genome_train[i])))
        for i in range(0, len(kmers_by_genome_train)):
            if num_kmers[i] > min(num_kmers):
                kmers_by_genome_train[i] = kmers_by_genome_train[i][:min(num_kmers)]
        for i in range(0, len(kmers_by_genome_train)):
            num_kmers[i] = len(kmers_by_genome_train[i])
            print("# k-mers in genome {0}: {1}".format(i, len(kmers_by_genome_train[i])))
        for i in range(0, len(kmers_by_genome_train)):
            test_num = int(.1 * len(kmers_by_genome_train[i]))
            kmers_by_genome_test.append(kmers_by_genome_train[i][:test_num])
            kmers_by_genome_train[i] = kmers_by_genome_train[i][test_num:]
        return SeqDatasetGenerator(kmers_by_genome_train, kmers_by_genome_test), file_list

    def save_to_file(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump([self.kmers_by_genome_train, self.kmers_by_genome_test], f)
        f.close()

    @staticmethod
    def load_from_file(file_name):
        f = open(file_name, 'rb')
        kmers_list = pickle.load(f)
        f.close()
        return SeqDatasetGenerator(kmers_by_genome_train=kmers_list[0], kmers_by_genome_test=kmers_list[1])

    @staticmethod
    def __make_vec(xs):
        return np.array(list(map(seq2binary, xs)))

    @staticmethod
    def __generate_data(batch_size, kmers):
        xs = []
        xps = []
        xns = []
        for i in range(0, batch_size):
            p, n = tuple(sample(range(0, len(kmers)), 2))
            x, xp = tuple(sample(kmers[p], 2))
            xn, = tuple(sample(kmers[n], 1))
            xs.append(x)
            xps.append(xp)
            xns.append(xn)
        xs  = SeqDatasetGenerator.__make_vec(xs)
        xps = SeqDatasetGenerator.__make_vec(xps)
        xns = SeqDatasetGenerator.__make_vec(xns)
        return xs, xps, xns

    @staticmethod
    def __generate_data_one_genome(batch_size, kmers, a_num, b_num):
        xs = []
        xps = []
        xns = []
        for i in range(0, batch_size):
            p, n = (a_num, b_num)
            x, xp = tuple(sample(kmers[p], 2))
            xn, = tuple(sample(kmers[n], 1))
            xs.append(x)
            xps.append(xp)
            xns.append(xn)
        xs = SeqDatasetGenerator.__make_vec(xs)
        xps = SeqDatasetGenerator.__make_vec(xps)
        xns = SeqDatasetGenerator.__make_vec(xns)
        return xs, xps, xns

    def generate_data_train(self, batch_size):
        return SeqDatasetGenerator.__generate_data(batch_size, self.kmers_by_genome_train)

    def generate_data_test(self, batch_size, a_num, b_num):
        return SeqDatasetGenerator.__generate_data_one_genome(batch_size, self.kmers_by_genome_test, a_num, b_num)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable("weights", dtype=tf.float32, initializer=initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.get_variable("biases", dtype=tf.float32, initializer=initial)


def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def compute_euclidean_distances(x, y, w=None):
    d = tf.square(tf.subtract(x, y))
    if w is not None:
        d = tf.transpose(tf.multiply(tf.transpose(d), w))
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class Triplet:
    def __init__(self, kmer_len, alpha):

        self.kmer_len = kmer_len
        self.alpha = alpha

        # Input and label placeholders
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='x')
            self.xp = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xp')
            self.xn = tf.placeholder(tf.float32, shape=[None, kmer_len, seq_dim], name='xn')
            self.pos_weights = tf.placeholder_with_default(shape=[None], name='pos_weights', input=np.array([]))
            self.neg_weights = tf.placeholder_with_default(shape=[None], name='neg_weights', input=np.array([]))
            self.top_k = tf.placeholder_with_default(shape=[], name='top_k_losses', input=75)

        with tf.variable_scope('embedding') as scope:
            self.o = self.__embedding_network(self.x, kmer_len) # self.o is the embedded genome
            scope.reuse_variables()
            self.op = self.__embedding_network(self.xp, kmer_len)
            self.on = self.__embedding_network(self.xn, kmer_len)

        with tf.variable_scope('distances'):
            self.dp = compute_euclidean_distances(self.o, self.op)
            self.dn = compute_euclidean_distances(self.o, self.on)
            #             self.dp = compute_euclidean_distances(self.o, self.op, None)
            #             self.dn = compute_euclidean_distances(self.o, self.on, None)
            # softmax creates a ratio measure of one distance to the other (ideally would be 0)
            # self.logits = tf.nn.softmax([self.dp, self.dn], name="logits")

        with tf.variable_scope('loss'):
            self.loss = tf.nn.relu(tf.pow(self.dp, 2) - tf.pow(self.dn, 2) + alpha)  # alpha is margin
            # self.loss = self.loss + (tf.norm(self.o) + tf.norm(self.op) + tf.norm(self.on))*0.00001  # regularization
            # train only on top k easiest values after a certain point
            self.loss = -tf.reduce_mean(tf.nn.top_k(-self.loss, k=self.top_k).values)

    def __embedding_network(self, x, kmer_len):

        dim = seq_dim
        with tf.variable_scope('conv1'):
            out = 32
            w = weight_variable([3, dim, out])
            b = bias_variable([out])
            # conv1d multiplies sections of input by weights w and returns feature map
            # relu introduces non-linearity in the convolutions
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
            # pooling downsizes data and prevents over-fitting
            gpool = tf.nn.pool(x, [h.get_shape()[1]], pooling_type="AVG", padding="VALID", name="gpool")
            # return tf.reshape(gpool, [-1, 128])
            return tf.reshape(gpool, [-1, 128])  # reshapes tensor to 1D x 128


# set parameters
k_mer_len = 100
batch_size = 200
logging_frequency = 25
iterations = 2000
margin = 1
visualization_batch_size = 100
top_k_iter_start = 300  # after how many iterations training on only easiest triplets should begin
n_easiest = batch_size  # how many triplets from the batch to calculate loss by after top_k_iter_start reached
seq_dim = 4

print("...loading k-mers from genome files")
try:
    seq_dataset_generator = SeqDatasetGenerator.load_from_file(file_name=str(k_mer_len)+'seq_dataset')
    file_list = SeqDatasetGenerator.get_file_list()
except FileNotFoundError:
    seq_dataset_generator, file_list = SeqDatasetGenerator.default_load(k=k_mer_len)
    seq_dataset_generator.save_to_file(file_name=str(k_mer_len)+'seq_dataset')

triplet = Triplet(kmer_len=k_mer_len, alpha=margin)
train_step = tf.train.AdamOptimizer(10e-5).minimize(triplet.loss)

print("...training model")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        top_k = batch_size if i < top_k_iter_start else n_easiest
        # top_k = batch_size
        batch = seq_dataset_generator.generate_data_train(batch_size)
        if i % logging_frequency == 0:
            loss = sess.run(triplet.loss, feed_dict={triplet.top_k: n_easiest, triplet.x: batch[0], triplet.xp: batch[1], triplet.xn: batch[2]})
            print('step %d, training loss %g' % (i, loss))
        train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1], triplet.xn: batch[2]})

    print("...validating model")
    embed_array = []
    confusion_matrix = np.zeros((len(file_list), len(file_list)))
    distance_matrix = np.zeros((len(file_list), len(file_list)))
    mean_vecs = []
    vars = []

    # test model, print loss when every genome compared against every other
    top_k = visualization_batch_size  # calculate test loss on all triplets, not just easiest
    for a_num in range(0, len(file_list)):
        a_embeddings = 0
        for b_num in range(0, len(file_list)):
            # print("a, b: {},{}".format(a_num, b_num))
            if a_num == b_num:
                confusion_matrix[a_num, b_num] = float('nan')
            else:
                # a_name = file_list[a_name]
                # b_name = file_list[b_name]

                test_batch = seq_dataset_generator.generate_data_test(visualization_batch_size, a_num, b_num)
                test_loss = sess.run(triplet.loss,
                                     feed_dict={triplet.x: test_batch[0], triplet.xp: test_batch[1], triplet.xn: test_batch[2]})
                confusion_matrix[a_num, b_num] = test_loss

                try:
                    a_embeddings = np.append(a_embeddings, sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]}), axis=0)
                except ValueError:
                    a_embeddings = sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]})

        # print("a_embedding shape: {}".format(a_embeddings.shape))
        embed_array.append(a_embeddings)
        mean = np.mean(embed_array[a_num], axis=0)
        mean_vecs.append(mean)
        dist_from_mean = embed_array[a_num] - mean
        sq_dist = np.square(dist_from_mean)
        mag_dist = np.linalg.norm(sq_dist, axis=1)
        var = sqrt(sum(mag_dist))
        vars.append(var)

    # print("embed array length: {}".format(len(embed_array)))
    # for i in range(0, len(embed_array)):
        # print("embed array entry {} shape: {}".format(i, embed_array[i].shape))

    for a_num in range(0, len(file_list)):
        for b_num in range(0, len(file_list)):
            distance_matrix[a_num, b_num] = np.linalg.norm(mean_vecs[a_num] - mean_vecs[b_num])

    print("variances")
    print(vars)
    print("distance matrix")
    print(distance_matrix)
    print("confusion matrix of losses")
    print(confusion_matrix)

    print("...graphing test embeddings")
    flat_list = [item for sublist in embed_array for item in sublist]

    tsne_model = TSNE(n_components=2, verbose=0)
    Y = tsne_model.fit_transform(flat_list)

    cmap = get_cmap(len(file_list)+1)

    for a_num in range(0, len(file_list)):
        start = visualization_batch_size*a_num*(len(file_list) - 1)
        stop = start + visualization_batch_size*(len(file_list) - 1)
        plt.scatter(Y[start:stop, 0], Y[start:stop, 1], c=cmap(a_num), label=file_list[a_num])
    plt.legend()
    plt.title("k=" + str(k_mer_len) + " iter=" + str(iterations) +
              " batch_size=" + str(batch_size) + " embed_dim=" + str(128))  # RECORD DIMENSION HERE

    os.system('say "finished"')
    plt.savefig("Figures/triplet_figure")
    # plt.show()


