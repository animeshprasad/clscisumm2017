#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author : Animesh Prasad
#Copyright : WING - NUS


from __future__ import print_function
import os
import numpy as np
import math
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, LSTM, Dropout, Activation, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional, Merge, TimeDistributed
from keras.models import Model
from keras.metrics import binary_crossentropy 
import sys
import read_data
from keras.optimizers import RMSprop, SGD

#import tensorflow as tf
#tf.python.control_flow_ops = tf

from sklearn.metrics import precision_recall_fscore_support

GLOVE_DIR = '../../glove'
MAX_SEQUENCE_LENGTH = 100
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



texts_r, texts_c, facets = read_data.get_data('1b')
t_texts_r, t_texts_c, others  = read_data.get_data('test1b')
print('Found {} samples with label.'.format(len(texts_c)))

print('Found {} samples without label.'.format(len(t_texts_c)))

class_ratio=np.bincount(facets)
class_weight={}
for i in xrange(len(set(facets))):
    class_weight[i] = math.log(np.sum(class_ratio) / class_ratio[i], 100) + 1
print('class ratio is : {}'.format(class_ratio))
print('using class weight of inverse ratio as : {}'.format(class_weight))



# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=True)
# Fix for already tokenized texts
tokenizer.fit_on_texts(texts_r + texts_c +t_texts_r + t_texts_c)
sequences1 = tokenizer.texts_to_sequences(texts_r)
sequences2 = tokenizer.texts_to_sequences(texts_c)

t_sequences1 = tokenizer.texts_to_sequences(t_texts_r)
t_sequences2 = tokenizer.texts_to_sequences(t_texts_c)

print (sequences1[0], sequences2[0], texts_r[0], texts_c[0])
#verify word indicies to be correct

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

t_data1 = pad_sequences(t_sequences1, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
t_data2 = pad_sequences(t_sequences2, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

labels = to_categorical(np.asarray(facets))

print('Shape of ref tensor:', data1.shape)
print('Shape of cit tensor:', data2.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data1.shape[0])
np.random.shuffle(indices)
data1 = data1[indices]
data2 = data2[indices]
labels = labels[indices]


# split the data into a training set and a validation set
nb_validation_samples = int(VALIDATION_SPLIT * data1.shape[0])
data1_v = data1[-nb_validation_samples:]
data2_v = data2[-nb_validation_samples:]
labels_v = labels[-nb_validation_samples:]

#used for Parameter Tuning

x1_train = data1[:-nb_validation_samples]
x2_train = data2[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

x1_val = data1_v
x2_val = data2_v
y_val = labels_v

x1_test = t_data1
x2_test = t_data2


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


#left = Sequential()
#left.add(embedding_layer1)
#left.add(LSTM(output_dim=100, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),return_sequences=False))

#right = Sequential()
#right.add(embedding_layer2)
#right.add(LSTM(output_dim=100, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),return_sequences=False))

#model = Sequential()
#model.add(Merge([left, right],mode='dot'))
#model.add(Dense(5))
#model.add(Dropout(0.4))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy',optimizer=RMSprop(clipvalue=0.99))

#model.fit([x1_train,x2_train], y_train, nb_epoch=2,
#          validation_data=([x1_val, x2_val], y_val),class_weight=class_weight)


left_c = Sequential()
left_c.add(embedding_layer1)
left_c.add(Conv1D(128, 5, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM), activation='relu'))
#left_c.add(MaxPooling1D(25))
#left_c.add(Conv1D(32, 5, activation='relu'))
left_c.add(GlobalMaxPooling1D())
left_c.add(Dense(100))
left_c.add(Dropout(0.2))

right_c = Sequential()
right_c.add(embedding_layer2)
right_c.add(Conv1D(128, 5, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM), activation='relu'))
#right_c.add(MaxPooling1D(25))
#right_c.add(Conv1D(32, 5, activation='relu'))
right_c.add(GlobalMaxPooling1D())
right_c.add(Dense(100))
right_c.add(Dropout(0.2))

model = Sequential()
model.add(Merge([left_c, right_c],mode='dot'))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.001,clipvalue=0.99))

model.fit([x1_train,x2_train], y_train, nb_epoch=6,
          validation_data=([x1_val, x2_val], y_val),class_weight=class_weight)



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


facet_types = ['Hypothesis_Citation', 'Aim_Citation',
               'Implication_Citation', 'Results_Citation', 'Method_Citation']


print ('Predict')
k=model.predict([x1_test, x2_test])
kj=[]
for i in xrange(len(k)):
    f=[]
    f.append(facet_types[k[i].argmax()])
    thres =k[i][k[i].argmax()] -0.01
    for j in xrange(5):
        if k[i][j] > thres and j != k[i].argmax():
            #print(facet_types[j],)
            f.append(facet_types[j])
    #print(facet_types[k[i].argmax()],)

    kj.append(f)
    print (f)
print ('True')



import os
folders = os.listdir('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-2017/Task1/')
for files in folders:
    f=open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-2017/Task1/'+files,'r')
    f2 = open('/Users/animeshprasad/PycharmProjects/scisumm/data/wing/Task1/' + files, 'w')
    for lines in f:
        print(lines)
        #print(others)
        flag=0
        for i,c in enumerate(others):
            c=c.split('|')
            if c[0].strip() in lines and c[1].strip() in lines:
                print (lines)
                flag =1
                print('yas')
                break
        if flag ==0:
            print("Error")
            print (lines)
            raw_input()
        f2.write(lines.strip()+' ')
        f2.write(str(kj[i]))
        f2.write('\n')
for i in xrange(len(others)):
    print(others[i], kj[i],t_texts_c[i])









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
