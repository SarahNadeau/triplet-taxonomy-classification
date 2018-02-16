
# adapted from the following: https://keras.io/getting-started/functional-api-guide/
# with input from http://localhost:8972/notebooks/triplet_keras.ipynb

import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense
from keras.layers import Embedding, Flatten, Input, merge
from keras.models import Model
import os
import pickle
import numpy as np

seq_x = Input(shape=(150, 4))
seq_p = Input(shape=(150, 4))
seq_n = Input(shape=(150, 4))

# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_x = shared_lstm(seq_x)
encoded_p = shared_lstm(seq_p)
encoded_n = shared_lstm(seq_n)

# Then calculate the loss
def triplet_loss(encoded_x, encoded_p, encoded_n):
    loss = 1 - K.sigmoid(
        K.sum(encoded_x * encoded_p, axis=1, keepdims=True) -
        K.sum(encoded_x * encoded_n, axis=1, keepdims=True))
    return loss

def build_model():
    model.add(Conv1D(32, 5, padding='same', input_shape=x_train.shape[1:]))



batch_size = 4 ### was 32
num_classes = 6 ### need to change if more genomes added
epochs = 2 ### 100 --> 50% accuracy
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_triplet_trained_model.h5'


# binarize an array of DNA sequences
# takes in a 1D array of DNA sequences, returns a 2D array of binarized DNA sequences
def seq2binary(seqs):
    ### can't handle difference sequence lengths
    seq_vecs = np.empty((len(seqs), len(seqs[0]), 4), dtype=int)
    # print('seq_vecs shape: {}'.format(seq_vecs.shape))
    for i in range(0, len(seqs)):
        ### eliminates N bases
        for j in range(0, len(seqs[i])):
            if seqs[i][j] == 'A':
                seq_vecs[i][j][0] = 1
            elif seqs[i][j] == 'C':
                seq_vecs[i][j][1] = 1
            elif seqs[i][j] == 'T':
                seq_vecs[i][j][2] = 1
            elif seqs[i][j] == 'G':
                seq_vecs[i][j][3] = 1
    return seq_vecs


# convert a list of string y labels to a list of unique ints starting at 0
def enumerate_y_labels(y_str):
    ylabel_dict = dict([(y, x) for x, y in enumerate(set(sorted(y_str)))])
    # print(y_str)
    # print([ylabel_dict[x] for x in y_str])
    return [ylabel_dict[x] for x in y_str]


# compute euclidean distances between two sample embeddings
def compute_euclidean_distances(x, y, w=None):
    d = tf.square(tf.subtract(x, y))
    if w is not None:
        d = tf.transpose(tf.multiply(tf.transpose(d), w))
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d


class DatasetGenerator(object):

    def __init__(self, x_in, y_in):
        self.x = x_in
        self.y = y_in

    # this function chooses reference samples (xs), positive samples (xps), and negative samples (xns) from the
    # reference k-mer lists
    @staticmethod
    def data_generator(batch_size, sample_dict):
        keep_going = True
        while keep_going:
            p_class, n_class = np.random.choice(range(0, num_classes), 2, replace=False)
            x_ind, p_ind = np.random.choice(range(0, len(sample_dict[p_class])), 2, replace=False)
            x = sample_dict[p_class][x_ind]
            xp = sample_dict[p_class][p_ind]
            n_ind = np.random.choice(range(0, len(sample_dict[n_class])))
            xn = sample_dict[n_class][n_ind]
            yield x, xp, xn


# The data, shuffled and split between train and test sets:
with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_test150.pickle', 'rb') as f:
    x_test = pickle.load(f)
    x_test = seq2binary(x_test)
    f.close()
with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/X_train150.pickle', 'rb') as f:
    x_train = pickle.load(f)
    x_train = seq2binary(x_train)
    f.close()
with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_test150.pickle', 'rb') as f:
    y_test_str = pickle.load(f)
    y_test = enumerate_y_labels(y_test_str)
    f.close()
with open('/Users/nadeau/Documents/Metagenome_Classification/train_test_set/y_train150.pickle', 'rb') as f:
    y_train_str = pickle.load(f)
    y_train = enumerate_y_labels(y_train_str)
    f.close()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train_1D = y_train
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

### attempting to mimic triplet paper
# model = Sequential()
# ### 32 is num filters, 3 is window size
# model.add(Conv1D(32, 5, padding='same', input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(64, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(128, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(256, 2))
# model.add(Flatten()) ### This isn't in the triplet paper, but they did have final embedding in 128D
# # model.add(Dropout(0.25))

model = Sequential()
### 32 is num filters, 3 is window size
model.add(Conv1D(32, 3, padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('Not using data augmentation.')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(datagen.flow(x_train, y_train,
    #                                  batch_size=batch_size),
    #                     epochs=epochs,
    #                     validation_data=(x_test, y_test),
    #                     workers=4)

# Fit the model on batches generated by me
x_train_dict = {} # keys are integer y_train labels, values are list of corresponding x_train samples
for i in range(0, len(x_train)):
    if y_train_1D[i] not in x_train_dict:
        x_train_dict[y_train_1D[i]] = [x_train[i]]
    else:
        x_train_dict[y_train_1D[i]].append(x_train[i])

model.fit_generator(DatasetGenerator.data_generator(batch_size,
                    sample_dict=x_train_dict), epochs=epochs, steps_per_epoch=int(x_train.shape[0]/batch_size))

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