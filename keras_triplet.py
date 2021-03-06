
# adapted from the following: https://keras.io/getting-started/functional-api-guide/
# with input from http://localhost:8972/notebooks/triplet_keras.ipynb
# triplet model implementation: https://github.com/maciejkula/triplet_recommendations_keras

import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Embedding, Flatten, Input, merge
from keras.models import Model
import os
import pickle
import numpy as np
from keras.preprocessing.text import one_hot

# seq_x = Input(shape=(150, 4))
# seq_p = Input(shape=(150, 4))
# seq_n = Input(shape=(150, 4))
#
# # This layer can take as input a matrix
# # and will return a vector of size 64
# shared_lstm = LSTM(64)
#
# # When we reuse the same layer instance
# # multiple times, the weights of the layer
# # are also being reused
# # (it is effectively *the same* layer)
# encoded_x = shared_lstm(seq_x)
# encoded_p = shared_lstm(seq_p)
# encoded_n = shared_lstm(seq_n)
#
# # Then calculate the loss
# def triplet_loss(encoded_x, encoded_p, encoded_n):
#     loss = 1 - K.sigmoid(
#         K.sum(encoded_x * encoded_p, axis=1, keepdims=True) -
#         K.sum(encoded_x * encoded_n, axis=1, keepdims=True))
#     return loss

# def build_model():
#     model.add(Conv1D(32, 5, padding='same', input_shape=x_train.shape[1:]))

batch_size = 4
num_classes = 6 ### need to change if more genomes added
epochs = 2
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_triplet_trained_model.h5'
vocab_size = 10 ### why 10 necessary to get A,C,T,G,N all hashed differently?
embed_dim = 2   ### should choose more appropriate dimension?
# binarize an array of DNA sequences
# takes in a 1D array of DNA sequences, returns a 2D array of binarized DNA sequences
# def seq2binary(seqs):
#     ### can't handle difference sequence lengths
#     seq_vecs = np.empty((len(seqs), len(seqs[0]), 4), dtype=int)
#     # print('seq_vecs shape: {}'.format(seq_vecs.shape))
#     for i in range(0, len(seqs)):
#         ### eliminates N bases
#         for j in range(0, len(seqs[i])):
#             if seqs[i][j] == 'A':
#                 seq_vecs[i][j][0] = 1
#             elif seqs[i][j] == 'C':
#                 seq_vecs[i][j][1] = 1
#             elif seqs[i][j] == 'T':
#                 seq_vecs[i][j][2] = 1
#             elif seqs[i][j] == 'G':
#                 seq_vecs[i][j][3] = 1
#     return seq_vecs


# convert a list of string y labels to a list of unique ints starting at 0

# translate string data labels to ints
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(set(sorted(y_str)))])
    return [ylabel_dict[x] for x in y_str]

# # compute euclidean distances between two sample embeddings
# def compute_euclidean_distances(x, y, w=None):
#     d = tf.square(tf.subtract(x, y))
#     if w is not None:
#         d = tf.transpose(tf.multiply(tf.transpose(d), w))
#     d = tf.sqrt(tf.reduce_sum(d, axis=1))
#     return d

class DatasetGenerator(object):

    def __init__(self, x_in, y_in):
        self.x = x_in
        self.y = y_in

    # sample reference seq (xs), positive seq (xps), and negative seq (xns)
    @staticmethod
    def get_triplet(sample_dict):
        p_class, n_class = np.random.choice(range(0, num_classes), 2, replace=False)
        x_ind, p_ind = np.random.choice(range(0, len(sample_dict[p_class])), 2, replace=False)
        x = sample_dict[p_class][x_ind]
        xp = sample_dict[p_class][p_ind]
        n_ind = np.random.choice(range(0, len(sample_dict[n_class])))
        xn = sample_dict[n_class][n_ind]
        return x, xp, xn


# load data
def load_data(vocab_size, num_classes):
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_test150.pickle', 'rb') as f:
        x_test = pickle.load(f)
        # make sequences into sentances of words
        # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
        x_test = [[letter for letter in word] for word in x_test]
        x_test = [" ".join(letters) for letters in x_test]
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_train150.pickle', 'rb') as f:
        x_train = pickle.load(f)
        x_train = [[letter for letter in word] for word in x_train]
        x_train = [" ".join(letters) for letters in x_train]
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_test150.pickle', 'rb') as f:
        y_test_str = pickle.load(f)
        y_test = enumerate_y_labels(y_test_str)
        f.close()
    with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_train150.pickle', 'rb') as f:
        y_train_str = pickle.load(f)
        y_train = enumerate_y_labels(y_train_str)
        f.close()

    # integer encode the "words" in sequences
    x_test = [[one_hot(s, vocab_size)] for s in x_test]
    x_test = np.array(x_test)
    x_train = [[one_hot(s, vocab_size)] for s in x_train]
    x_train = np.array(x_train)
    print('x train shape: {}'.format(x_train.shape))

    # covert int y label vectors to one hot matrices
    y_test_1D = y_test
    y_train_1D = y_train
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y train shape: {}'.format(y_train.shape))

    return y_train_1D, y_test_1D, x_train, y_train, x_test, y_test


# get identity loss
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


# get triplet loss
def bpr_triplet_loss(X):
    xp, xn, x = X
    loss = 1.0 - K.sigmoid(
        K.sum(x * xp, axis=-1, keepdims=True) -
        K.sum(x * xn, axis=-1, keepdims=True))
    return loss

def build_model(vocab_size, embed_dim):
    xp_input = Input((150, ), name='xp_input')
    xn_input = Input((150, ), name='xn_input')
    x_input = Input((150, ), name='x_input')

    embedding_layer = Embedding(
        vocab_size, embed_dim, input_length=150)

    xp_embedding = Flatten()(embedding_layer(xp_input))
    xn_embedding = Flatten()(embedding_layer(xn_input))
    x_embedding = Flatten()(embedding_layer(x_input))

    loss = merge(
        [xp_embedding, xn_embedding, x_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[xp_input, xn_input, x_input],
        output=loss)

    # compile the model
    model.compile(loss=identity_loss, optimizer=Adam())

    return model

# summarize the model
# print(model.summary())

# fit the model
# output_array = model.predict(x_train[0])
# assert output_array.shape == (1, 150, 2)

### MAIN STARTS HERE ###

# load da

y_train_1D, y_test_1D, x_train, y_train, x_test, y_test = load_data(vocab_size, num_classes)

# # sample triplet from testing data
# x_test_dict = {} # keys are integer y_train labels, values are list of corresponding x_train samples
# for i in range(0, len(x_test)):
#     if y_test_1D[i] not in x_test_dict:
#         x_test_dict[y_test_1D[i]] = [x_test[i]]
#     else:
#         x_test_dict[y_test_1D[i]].append(x_test[i])
# test_x, test_xp, test_xn = DatasetGenerator.get_triplet(x_test_dict)

model = build_model(vocab_size, embed_dim)
print(model.summary())

# sample triplet from training data
x_train_dict = {} # keys are integer y_train labels, values are list of corresponding x_train samples
for i in range(0, len(x_train)):
    if y_train_1D[i] not in x_train_dict:
        x_train_dict[y_train_1D[i]] = [x_train[i]]
    else:
        x_train_dict[y_train_1D[i]].append(x_train[i])
x, xp, xn = DatasetGenerator.get_triplet(x_train_dict)

X = {
    'x_input': x,
    'xp_input': xp,
    'xn_input': xn
}

model.fit(X, np.ones(len(x)), batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

# evaluate the model
# loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
# print('Accuracy: %f' % (accuracy*100))

# # Fit the model on batches generated by me
# x_train_dict = {} # keys are integer y_train labels, values are list of corresponding x_train samples
# for i in range(0, len(x_train)):
#     if y_train_1D[i] not in x_train_dict:
#         x_train_dict[y_train_1D[i]] = [x_train[i]]
#     else:
#         x_train_dict[y_train_1D[i]].append(x_train[i])
#
# model.fit_generator(DatasetGenerator.data_generator(batch_size,
#                     sample_dict=x_train_dict), epochs=epochs, steps_per_epoch=int(x_train.shape[0]/batch_size))

# Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
#
# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

