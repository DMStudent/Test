# Michael A. Alcorn (malcorn@redhat.com)
# A (slightly modified) implementation of the Recurrent Convolutional Neural Network (RCNN) found in [1].
# [1] Siwei, L., Xu, L., Kang, L., and Zhao, J. 2015. Recurrent convolutional
#         neural networks for text classification. In AAAI, pp. 2267-2273.
#         http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

from keras import backend
from keras.layers import Input, Lambda, LSTM, TimeDistributed, Dense, Embedding
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup
import os


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('../textClassifier/data/imdb/labeledTrainData.tsv', sep='\t')
print data_train.shape

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
    labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set ')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

GLOVE_DIR = "/search/odin/data/wangyuan/pycharmProjects/keras_demo/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

hidden_dim_1 = 200
hidden_dim_2 = 100
NUM_CLASSES = 2


document = Input(shape = (MAX_SEQUENCE_LENGTH, ), dtype = "int32")
#left_context = Input(shape = (None, ), dtype = "int32")
#right_context = Input(shape = (None, ), dtype = "int32")

doc_embedding = embedding_layer(document)
l_embedding = embedding_layer(document)
r_embedding = embedding_layer(document)

# I use LSTM RNNs instead of vanilla RNNs as described in the paper.
forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding) # See equation (1).
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).

# Keras provides its own max-pooling layers, but they cannot handle variable length input
# (as far as I can tell). As a result, I define my own max-pooling layer here.
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) # See equations (6) and (7).

model = Model(inputs = [document], outputs = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])
print model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=50)
model.save("model/rcnn.model")
# text = "This is some example text."
# text = text.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
# tokens = text.split()
# tokens = [word2vec.vocab[token].index if token in word2vec.vocab else MAX_NB_WORDS for token in tokens]
#
# doc_as_array = np.array([tokens])
# # We shift the document to the right to obtain the left-side contexts.
# left_context_as_array = np.array([[MAX_NB_WORDS] + tokens[:-1]])
# # We shift the document to the left to obtain the right-side contexts.
# right_context_as_array = np.array([tokens[1:] + [MAX_NB_WORDS]])
#
# target = np.array([NUM_CLASSES * [0]])
# target[0][3] = 1
#
# history = model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, epochs = 1, verbose = 0)
# loss = history.history["loss"][0]











# word2vec = gensim.models.Word2Vec.load("word2vec.gensim")
# # We add an additional row of zeros to the embeddings matrix to represent unseen words and the NULL token.
# embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype = "float32")
# embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
#
# MAX_TOKENS = word2vec.syn0.shape[0]
# embedding_dim = word2vec.syn0.shape[1]
# hidden_dim_1 = 200
# hidden_dim_2 = 100
# NUM_CLASSES = 10
#
# document = Input(shape = (None, ), dtype = "int32")
# left_context = Input(shape = (None, ), dtype = "int32")
# right_context = Input(shape = (None, ), dtype = "int32")
#
# embedder = Embedding(MAX_TOKENS + 1, embedding_dim, weights = [embeddings], trainable = False)
# doc_embedding = embedder(document)
# l_embedding = embedder(left_context)
# r_embedding = embedder(right_context)
#
# # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
# forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding) # See equation (1).
# backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
# together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).
#
# semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).
#
# # Keras provides its own max-pooling layers, but they cannot handle variable length input
# # (as far as I can tell). As a result, I define my own max-pooling layer here.
# pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).
#
# output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) # See equations (6) and (7).
#
# model = Model(inputs = [document, left_context, right_context], outputs = output)
# model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])
#
# text = "This is some example text."
# text = text.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
# tokens = text.split()
# tokens = [word2vec.vocab[token].index if token in word2vec.vocab else MAX_TOKENS for token in tokens]
#
# doc_as_array = np.array([tokens])
# # We shift the document to the right to obtain the left-side contexts.
# left_context_as_array = np.array([[MAX_TOKENS] + tokens[:-1]])
# # We shift the document to the left to obtain the right-side contexts.
# right_context_as_array = np.array([tokens[1:] + [MAX_TOKENS]])
#
# target = np.array([NUM_CLASSES * [0]])
# target[0][3] = 1
#
# history = model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, epochs = 1, verbose = 0)
# loss = history.history["loss"][0]
