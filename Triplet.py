import tensorflow as tf
import numpy as np
import os
from Bio import SeqIO
from random import shuffle, sample
from sklearn.manifold import TSNE
import _pickle as pickle
from math import sqrt
import sys
from get_test_reads_sra import get_test_reads, get_test_tax
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug


# this function binarizes DNA sequences (takes in sequence)
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

class DatasetGenerator(object):

    # def __init__(self, x_in, y_in):
    #     self.x = x_in
    #     self.y = y_in

    # this function binarizes sample k-mers
    # @staticmethod
    # def make_vec(xs):
    #     return np.array(list(map(seq2binary, xs)))

    # sample reference seq (xs), positive seq (xps), and negative seq (xns)
    @staticmethod
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

# class SeqDatasetGenerator(object):
#
#     def __init__(self, kmers_train, kmers_test):
#         self.kmers_train = kmers_train
#         self.kmers_test = kmers_test
#
#     # the function returns a list of Genbank sequence file names in the Genomes directory
#     @staticmethod
#     def get_file_list():
#         file_list = []
#         for file_name in os.listdir('Genomes'):
#             file_name = os.path.join("Genomes", file_name)
#             if file_name.endswith(".gbff"):
#                 file_list.append(file_name)
#         return file_list

    # this function generates lists of k-mers from reference sequences,
    # and partitions the k-mer lists into training and testing sets
    # @staticmethod
    # def default_load(k):
    #     kmers_train = []
    #     kmers_test = []
    #     num_kmers = []
    #     org_names = []
    #
    #     file_list = SeqDatasetGenerator.get_file_list()
    #
    #     for file in file_list:
    #         kmers = []
    #         for accession in SeqIO.parse(file, 'genbank'):
    #             print("genome {} has taxonomy {}".format(accession.id, accession.annotations["taxonomy"]))
    #             org_names.append(accession.id)
    #             seq = str(accession.seq)
    #             for s in range(0, len(seq) - k + 1):
    #                 kmers.append(seq[s:s + k])
    #         kmers_train.append(kmers)
    #
    #     for kmers in kmers_train:
    #         shuffle(kmers)
    #     for i in range(0, len(kmers_train)):
    #         num_kmers.append(len(kmers_train[i]))
    #         print("# k-mers in genome {0}: {1}".format(file_list[i], len(kmers_train[i])))
    #     for i in range(0, len(kmers_train)):
    #         if num_kmers[i] > min(num_kmers):
    #             kmers_train[i] = kmers_train[i][:min(num_kmers)]
    #     num_kmers = len(kmers_train[0])
    #     print("# k-mers in trimmed genomes: {0}".format(num_kmers))
    #
    #     for i in range(0, len(kmers_train)):
    #         test_num = int(.1 * len(kmers_train[i]))
    #         kmers_test.append(kmers_train[i][:test_num])
    #         kmers_train[i] = kmers_train[i][test_num:]
    #     return SeqDatasetGenerator(kmers_train, kmers_test), file_list, org_names
    #
    # # this function writes the k-mer lists to a text file
    # def save_to_file(self, file_name):
    #     f = open(file_name, 'wb')
    #     pickle.dump([self.kmers_train, self.kmers_test], f)
    #     f.close()

    # this function reads the k-mer lists from a text file
    # @staticmethod
    # def load_from_file(file_name):
    #     f = open(file_name, 'rb')
    #     kmers_list = pickle.load(f)
    #     f.close()
    #     return SeqDatasetGenerator(kmers_train=kmers_list[0], kmers_test=kmers_list[1])

    # # this function chooses reference samples (xs), positive samples (xps), and negative samples (xns) from the
    # # reference k-mer lists
    # @staticmethod
    # def __generate_data(batch_size, kmers):
    #     xs = []
    #     xps = []
    #     xns = []
    #     for i in range(0, batch_size):
    #         p, n = tuple(sample(range(0, len(kmers)), 2))
    #         x, xp = tuple(sample(kmers[p], 2))
    #         xn, = tuple(sample(kmers[n], 1))
    #         xs.append(x)
    #         xps.append(xp)
    #         xns.append(xn)
    #     xs  = SeqDatasetGenerator.__make_vec(xs)
    #     xps = SeqDatasetGenerator.__make_vec(xps)
    #     xns = SeqDatasetGenerator.__make_vec(xns)
    #     return xs, xps, xns


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


# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(set(sorted(y_str)))])
    return [ylabel_dict[x] for x in y_str]


# SET PARAMETERS HERE
k_mer_len = 150
batch_size = 40
logging_frequency = 10
iterations = 50
margin = 1
top_k_iter_start = 300
n_easiest = batch_size
seq_dim = 4
test_num = 10
embed_dim = 128

# load data
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

triplet = Triplet(kmer_len=k_mer_len, alpha=margin)
train_step = tf.train.AdamOptimizer(10e-5).minimize(triplet.loss)

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    print("...training  model")
    for i in range(iterations):
        top_k = batch_size if i < top_k_iter_start else n_easiest
        batch = DatasetGenerator.get_triplet_batch(x_train_dict, batch_size)
        # print(len(batch[0]))
        if i % logging_frequency == 0:
            loss = sess.run(triplet.loss, feed_dict={triplet.top_k: n_easiest, triplet.x: batch[0],
                                                     triplet.xp: batch[1], triplet.xn: batch[2]})
            print('step %d, training loss %g' % (i, loss))
        train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1],
                                  triplet.xn: batch[2]})

    print("...testing")
    # train_embed_array = []
    # train_mean_vecs = []
    # train_vars = []
    # embed_array = []
    # confusion_matrix = np.zeros((len(file_list), len(file_list)))
    # distance_matrix = np.zeros((len(file_list), len(file_list)))
    # mean_vecs = []
    # vars = []
    # unclassified_embed_array = []
    #
    # # embed unclassified reads
    # for file in unclassified_file_list:
    #     unclassified_reads = get_test_reads(file, k_mer_len, test_num)
    #     for i in range(0, len(unclassified_reads)):
    #         unclassified_reads[i] = seq2binary(unclassified_reads[i])
    #     unclassified_embeddings = sess.run(triplet.o, feed_dict={triplet.x: unclassified_reads})
    #     unclassified_embed_array.append(unclassified_embeddings)


    # # test model, print confusion matrix
    # for a_num in range(0, len(file_list)):
    #     a_embeddings = 0
    #     for b_num in range(0, len(file_list)):
    #         if a_num == b_num:
    #             confusion_matrix[a_num, b_num] = float('nan')
    #         else:
    #             test_batch = seq_dataset_generator.generate_data_test(visualization_batch_size, a_num, b_num)
    #             test_loss = sess.run(triplet.loss, feed_dict={triplet.x: test_batch[0], triplet.xp: test_batch[1],
    #                                                           triplet.xn: test_batch[2]})
    #             confusion_matrix[a_num, b_num] = test_loss
    #
    #             try:
    #                 a_embeddings = np.append(a_embeddings, sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]}),
    #                                          axis=0)
    #             except ValueError:
    #                 a_embeddings = sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]})
    #
    #     embed_array.append(a_embeddings)
    #     mean = np.mean(embed_array[a_num], axis=0)
    #     mean_vecs.append(mean)
    #     dist_from_mean = embed_array[a_num] - mean
    #     sq_dist = np.square(dist_from_mean)
    #     mag_dist = np.linalg.norm(sq_dist, axis=1)
    #     var = sqrt(sum(mag_dist))
    #     vars.append(var)

    # to label confusion matrix with genome names:
    # confusion_matrix = pd.DataFrame(confusion_matrix)
    # print(confusion_matrix)
    # confusion_matrix.columns = org_names
    # confusion_matrix.index = org_names

    # for a_num in range(0, len(file_list)):
    #     for b_num in range(0, len(file_list)):
    #         distance_matrix[a_num, b_num] = np.linalg.norm(mean_vecs[a_num] - mean_vecs[b_num])
    #
    # print("variance of embeddings for each genome:")
    # print(vars, '\n')
    # print("distances between mean embeddings of each genome:")
    # print(distance_matrix, '\n')
    # print("testing losses in comparisons between genomes:")
    # print(confusion_matrix)
    #
    # os.system('say "finished"')



