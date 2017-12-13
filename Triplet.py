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


class SeqDatasetGenerator(object):

    def __init__(self, kmers_train, kmers_test):
        self.kmers_train = kmers_train
        self.kmers_test = kmers_test

    @staticmethod
    def get_file_list():
        file_list = []
        for file_name in os.listdir('Genomes'):
            # print(file_name)
            file_name = os.path.join("Genomes", file_name)
            if file_name.endswith(".gbff"):
                file_list.append(file_name)
        return file_list

    @staticmethod
    def default_load(k):
        kmers_train = []
        kmers_test = []
        num_kmers = []
        org_names = []

        file_list = SeqDatasetGenerator.get_file_list()

        for file in file_list:
            kmers = []
            for accession in SeqIO.parse(file, 'genbank'):
                print("genome {} has taxonomy {}".format(accession.id, accession.annotations["taxonomy"]))
                org_names.append(accession.id)
                seq = str(accession.seq)
                for s in range(0, len(seq) - k + 1):
                    kmers.append(seq[s:s + k])
            kmers_train.append(kmers)

        for kmers in kmers_train:
            shuffle(kmers)
        for i in range(0, len(kmers_train)):
            num_kmers.append(len(kmers_train[i]))
            print("# k-mers in genome {0}: {1}".format(file_list[i], len(kmers_train[i])))
        for i in range(0, len(kmers_train)):
            if num_kmers[i] > min(num_kmers):
                kmers_train[i] = kmers_train[i][:min(num_kmers)]
        num_kmers = len(kmers_train[0])
        print("# k-mers in trimmed genomes: {0}".format(num_kmers))

        for i in range(0, len(kmers_train)):
            test_num = int(.1 * len(kmers_train[i]))
            kmers_test.append(kmers_train[i][:test_num])
            kmers_train[i] = kmers_train[i][test_num:]
        return SeqDatasetGenerator(kmers_train, kmers_test), file_list, org_names

    def save_to_file(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump([self.kmers_train, self.kmers_test], f)
        f.close()

    @staticmethod
    def load_from_file(file_name):
        f = open(file_name, 'rb')
        kmers_list = pickle.load(f)
        f.close()
        return SeqDatasetGenerator(kmers_train=kmers_list[0], kmers_test=kmers_list[1])

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
        return SeqDatasetGenerator.__generate_data(batch_size, self.kmers_train)

    def generate_data_visualization(self, batch_size, a_num, b_num):
        return SeqDatasetGenerator.__generate_data_one_genome(batch_size, self.kmers_train, a_num, b_num)

    def generate_data_test(self, batch_size, a_num, b_num):
        return SeqDatasetGenerator.__generate_data_one_genome(batch_size, self.kmers_test, a_num, b_num)


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

        with tf.variable_scope('loss'):
            self.loss = tf.nn.relu(tf.pow(self.dp, 2) - tf.pow(self.dn, 2) + alpha)
            self.loss = -tf.reduce_mean(tf.nn.top_k(-self.loss, k=self.top_k).values)

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


# set parameters
k_mer_len = 150
batch_size = 1000
logging_frequency = 25
iterations = 500
margin = 1
visualization_batch_size = 100
top_k_iter_start = 300
n_easiest = batch_size
seq_dim = 4
test_num = 10
embed_dim = 128

print("...loading k-mers from genome files")
try:
    seq_dataset_generator = SeqDatasetGenerator.load_from_file(file_name=str(k_mer_len)+'seq_dataset')
    file_list = SeqDatasetGenerator.get_file_list()
    for i in range(0, len(file_list)):
        file_list[i] = file_list[i].split('/')[1].split('.')[0]
    org_names = file_list
except FileNotFoundError:
    seq_dataset_generator, file_list, org_names = SeqDatasetGenerator.default_load(k=k_mer_len)
    seq_dataset_generator.save_to_file(file_name=str(k_mer_len)+'seq_dataset')

unclassified_file_list = []
for file_name in os.listdir('SRA_Test_Sequences'):
    if file_name.endswith(".fastq"):
        file_name = os.path.join('SRA_Test_Sequences', file_name)
        unclassified_file_list.append(file_name)

triplet = Triplet(kmer_len=k_mer_len, alpha=margin)
train_step = tf.train.AdamOptimizer(10e-5).minimize(triplet.loss)

print("...training model")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        top_k = batch_size if i < top_k_iter_start else n_easiest
        batch = seq_dataset_generator.generate_data_train(batch_size)
        if i % logging_frequency == 0:
            loss = sess.run(triplet.loss, feed_dict={triplet.top_k: n_easiest, triplet.x: batch[0],
                                                     triplet.xp: batch[1], triplet.xn: batch[2]})
            print('step %d, training loss %g' % (i, loss))
        train_step.run(feed_dict={triplet.top_k: top_k, triplet.x: batch[0], triplet.xp: batch[1],
                                  triplet.xn: batch[2]})

    print("...embedding unclassified k-mers")
    train_embed_array = []
    train_mean_vecs = []
    train_vars = []
    embed_array = []
    confusion_matrix = np.zeros((len(file_list), len(file_list)))
    distance_matrix = np.zeros((len(file_list), len(file_list)))
    mean_vecs = []
    vars = []
    unclassified_embed_array = []

    # embed training k-mers for plotting
    for a_num in range(0, len(file_list)):
        a_embeddings = 0
        for b_num in range(0, len(file_list)):
            if a_num != b_num:
                visualize_batch = seq_dataset_generator.generate_data_visualization(visualization_batch_size, a_num, b_num)
                try:
                    a_embeddings = np.append(a_embeddings, sess.run(triplet.o, feed_dict={triplet.x: visualize_batch[0]}), axis=0)
                except ValueError:
                    a_embeddings = sess.run(triplet.o, feed_dict={triplet.x: visualize_batch[0]})

        train_embed_array.append(a_embeddings)
        mean = np.mean(train_embed_array[a_num], axis=0)
        train_mean_vecs.append(mean)
        dist_from_mean = train_embed_array[a_num] - mean
        sq_dist = np.square(dist_from_mean)
        mag_dist = np.linalg.norm(sq_dist, axis=1)
        var = sqrt(sum(mag_dist))
        train_vars.append(var)

    # embed unclassified reads
    for file in unclassified_file_list:
        unclassified_reads = get_test_reads(file, k_mer_len, test_num)
        for i in range(0, len(unclassified_reads)):
            unclassified_reads[i] = seq2binary(unclassified_reads[i])
        unclassified_embeddings = sess.run(triplet.o, feed_dict={triplet.x: unclassified_reads})
        unclassified_embed_array.append(unclassified_embeddings)

    # plot unclassified reads with training set embeddings
    print("...creating figure \n")
    flat_list = [item for sublist in train_embed_array for item in sublist]
    flat_list = flat_list + [item for sublist in unclassified_embed_array for item in sublist]

    tsne_model = TSNE(n_components=2, verbose=0)
    Y = tsne_model.fit_transform(flat_list)

    cmap = get_cmap(len(unclassified_file_list) + len(file_list) + 2)

    for t in range(0, len(file_list)):
        start = visualization_batch_size * t * (len(file_list) - 1)
        stop = start + visualization_batch_size * (len(file_list) - 1)
        plt.scatter(Y[start:stop, 0], Y[start:stop, 1], c=cmap(len(unclassified_file_list) + 1 + t),
                    label=org_names[t])
        end_of_train = stop

    for u in range(0, len(unclassified_file_list)):
        start = test_num * u + end_of_train
        stop = start + test_num
        plt.scatter(Y[start:stop, 0], Y[start:stop, 1], c=cmap(u),
                    label=unclassified_file_list[u].split("/")[1].split(".")[0])

    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
    plt.legend(bbox_to_anchor=(0, -0.45), loc="lower left")
    plt.title("k=" + str(k_mer_len) + " iter=" + str(iterations) +
              " batch_size=" + str(batch_size) + " embed_dim=" + str(embed_dim))

    plt.savefig("Figures/train_and_unknown")

    # test model, print confusion matrix
    for a_num in range(0, len(file_list)):
        a_embeddings = 0
        for b_num in range(0, len(file_list)):
            if a_num == b_num:
                confusion_matrix[a_num, b_num] = float('nan')
            else:
                test_batch = seq_dataset_generator.generate_data_test(visualization_batch_size, a_num, b_num)
                test_loss = sess.run(triplet.loss, feed_dict={triplet.x: test_batch[0], triplet.xp: test_batch[1],
                                                              triplet.xn: test_batch[2]})
                confusion_matrix[a_num, b_num] = test_loss

                try:
                    a_embeddings = np.append(a_embeddings, sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]}),
                                             axis=0)
                except ValueError:
                    a_embeddings = sess.run(triplet.o, feed_dict={triplet.x: test_batch[0]})

        embed_array.append(a_embeddings)
        mean = np.mean(embed_array[a_num], axis=0)
        mean_vecs.append(mean)
        dist_from_mean = embed_array[a_num] - mean
        sq_dist = np.square(dist_from_mean)
        mag_dist = np.linalg.norm(sq_dist, axis=1)
        var = sqrt(sum(mag_dist))
        vars.append(var)

    # confusion_matrix = pd.DataFrame(confusion_matrix)
    # print(confusion_matrix)
    # confusion_matrix.columns = org_names
    # confusion_matrix.index = org_names

    for a_num in range(0, len(file_list)):
        for b_num in range(0, len(file_list)):
            distance_matrix[a_num, b_num] = np.linalg.norm(mean_vecs[a_num] - mean_vecs[b_num])

    print("variance of embeddings for each genome:")
    print(vars, '\n')
    print("distances between mean embeddings of each genome:")
    print(distance_matrix, '\n')
    print("testing losses in comparisons between genomes:")
    print(confusion_matrix)

    os.system('say "finished"')



