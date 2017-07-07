#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author : Animesh Prasad
#Copyright : WING - NUS


from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional, Merge, TimeDistributed
from keras.models import Model
from keras.metrics import binary_crossentropy 
import sys
import read_data
from keras.optimizers import RMSprop

#import tensorflow as tf
#tf.python.control_flow_ops = tf

from sklearn.metrics import precision_recall_fscore_support

GLOVE_DIR = '../../glove'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    #print(np.linalg.norm(coefs))
    embeddings_index[word] = coefs
f.close()



#for words in ['$PSELF', '$AREF', '$PREF', '$MENTION', '$SMENTION', '$CREF', '$EQN', '<unk>']:
    #k=np.random.rand(1, EMBEDDING_DIM)
    #k=7*k/np.linalg.norm(k)
    #embeddings_index[words] = k
    #print(np.linalg.norm(k))
#exit()


print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')


texts_r, texts_c, labels, flabels = read_data.get_data('1')
print('Found {} samples with label.'.format(len(texts_c)))


class_ratio=np.bincount(labels)
class_weight={}
for i in xrange(len(set(labels))):
    class_weight[i] = np.sum(class_ratio) / class_ratio[i]
print('class ratio is : {}'.format(class_ratio))
print('using class weight of inverse ratio as : {}'.format(class_weight))
#class_weight[1] = class_weight[1]*10000


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=True)
# Fix for already tokenized texts
tokenizer.fit_on_texts(texts_r + texts_c)
sequences1 = tokenizer.texts_to_sequences(texts_r)
sequences2 = tokenizer.texts_to_sequences(texts_c)

print (len(sequences1[0]), len(sequences2[0]), len(texts_r[0]), len(texts_c[0]))
#verify word indicies to be correct

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
data2 = pad_sequences(sequences2, maxlen=2*MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

labels = to_categorical(np.asarray(labels))

print('Shape of ref tensor:', data1.shape)
print('Shape of cit tensor:', data2.shape)
print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
indices = np.arange(data1.shape[0])
np.random.shuffle(indices)
data1 = data1[indices]
data2 = data2[indices]
labels = labels[indices]

#Used for Blind Dev Test
nb_validation_samples = int(VALIDATION_SPLIT * data1.shape[0])
data1_t = data1[:-nb_validation_samples]
data2_t = data2[:-nb_validation_samples]
labels_t = labels[:-nb_validation_samples]

#used for Parameter Tuning
nb_paramvalidation_samples = int(VALIDATION_SPLIT * data1_t.shape[0])

x1_train = data1_t[:-nb_paramvalidation_samples]
x2_train = data2_t[:-nb_paramvalidation_samples]
y_train = labels_t[:-nb_paramvalidation_samples]

x1_val = data1_t[-nb_paramvalidation_samples:]
x2_val = data2_t[-nb_paramvalidation_samples:]
y_val = labels_t[-nb_paramvalidation_samples:]

x1_test = data1[-nb_validation_samples:]
x2_test = data2[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]


print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
#print(word_index)
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        k=np.random.rand(1, EMBEDDING_DIM)
        k=7*k/np.linalg.norm(k)
        embedding_matrix[i] = k
	

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer1 = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

embedding_layer2 = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')


left = Sequential()
left.add(embedding_layer1)
left.add(LSTM(output_dim=100, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),return_sequences=False))
#left.add(TimeDistributed(Dense(100)))

right = Sequential()
right.add(embedding_layer2)
right.add(LSTM(output_dim=100, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),return_sequences=False))
#right.add(TimeDistributed(Dense(100)))

model = Sequential()
model.add(Merge([left, right],mode='dot'))
#model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(clipvalue=0.99))

model.fit([x1_train,x2_train], y_train, nb_epoch=2,
          validation_data=([x1_val, x2_val], y_val),class_weight=class_weight)


#model = Sequential()
#model.add(Merge([x1,x2], mode='dot'))
#model.add(Dense(2, activation='sigmoid')(x))





# train a 1D convnet with global maxpooling
#sequence1_input = Input(shape=(MAX_SEQUENCE_LENGTH,1), dtype='int32')
#sequence2_input = Input(shape=(MAX_SEQUENCE_LENGTH,1), dtype='int32')
#embedded1_sequences = embedding_layer(sequence1_input)
#embedded2_sequences = embedding_layer(sequence2_input)

#LSTM
#x1= Bidirectional(LSTM(64, dropout_W=0.2, dropout_U=0.2))(embedded1_sequences)
#x2= Bidirectional(LSTM(64, dropout_W=0.2, dropout_U=0.2))(embedded2_sequences)
#x1=Dense(64, activation='sigmoid')(x1)
#x2=Dense(64, activation='sigmoid')(x2)
#preds=Dense(2, activation='sigmoid')(x1)


#x= Merge([x1,x2], mode='dot')
#preds=Dense(2, activation='sigmoid')(x)

#x1 = Conv1D(128, 5, activation='relu')(embedded1_sequences)
#x2 = Conv1D(128, 5, activation='relu')(embedded2_sequences)
#x1 = MaxPooling1D(25)(x1)
#x2 = MaxPooling1D(25)(x2)
#x1 = Conv1D(32, 5, activation='relu')(x1)
#x1 = MaxPooling1D(25)(x1)
#x2 = Conv1D(32, 5, activation='relu')(x2)
#x2 = MaxPooling1D(25)(x2)
#x1 = Flatten()(x1)
#x2 = Flatten()(x2)
#x1 = Dropout(0.4)(Dense(64, activation='relu')(x1))
#x2 = Dropout(0.4)(Dense(64, activation='relu')(x2))
#x= Merge([x1,x2], mode='dot')
#preds = Dense(len(labels_index), activation='softmax')(x)

#model = Model([sequence1_input,sequence2_input], preds)
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['acc', binary_crossentropy])


#model.fit([x1_train,x2_train], y_train, validation_data=([x1_val, x2_val], y_val),
#          nb_epoch=2, batch_size=16, class_weight = None)





scores = model.evaluate([x1_test, x2_test], y_test, verbose=0)
print ('Predict')
print(  model.predict([x1_test, x2_test]).argmax(1))
print ('True')
print(  y_test.argmax(1))

a,b,c,d=precision_recall_fscore_support(model.predict([x1_test, x2_test]).argmax(1), y_test.argmax(1))
print (a[1],  b[1], c[1])



#from keras import backend as K

#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
#test = np.random.random(input_shape)[np.newaxis,...]
#for items in [#x_train, x_val, x_test]:
#    layer_outs = functor([x_test, 1.])
#    print(layer_outs)

#import theano
#get_activations = theano.function([model.layers[0].input], model.layers[1].output(train=False), allow_input_downcast=True)
#print (get_activations(x_train))
#print (get_activations(x_val))
#print (get_activations(x_test))



#print (a,b,c,d)
#print("%s: %.2f%%, %.2f, %.2f" % (model.metrics_names, scores[1]*100, scores[2], scores[3]))
