
# adapted from the following: https://keras.io/getting-started/sequential-model-guide/

'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
import os
import pickle
import numpy as np

batch_size = 32
num_classes = 6 ### need to change if more genomes added
epochs = 100 ### 100 --> 50% accuracy
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

class SeqDatasetGenerator(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    # this function chooses reference samples (xs), positive samples (xps), and negative samples (xns) from the
    # reference k-mer lists
    @staticmethod
    def __generate_data(batch_size, x, y):
        for (xi, yi) in x, y:
            print(xi, yi)
        # xs = []
        # xps = []
        # xns = []
        # for i in range(0, batch_size):
        #     p, n = tuple(sample(range(0, len(kmers)), 2))
        #     x, xp = tuple(sample(kmers[p], 2))
        #     xn, = tuple(sample(kmers[n], 1))
        #     xs.append(x)
        #     xps.append(xp)
        #     xns.append(xn)
        # xs  = SeqDatasetGenerator.__make_vec(xs)
        # xps = SeqDatasetGenerator.__make_vec(xps)
        # xns = SeqDatasetGenerator.__make_vec(xns)
        # return xs, xps, xns
        return


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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
### 32 is num filters, 3 is window size
model.add(Conv1D(32, 5, padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(256, 2))
model.add(MaxPooling1D(pool_size=2)) ### This isn't in the triplet paper, but they did have final embedding in 128D
# model.add(Dropout(0.25))

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

print(x_train)
print(x_test)


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

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
    model.fit_generator(SeqDatasetGenerator.__generate_data(x_train, y_train))

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