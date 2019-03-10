#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:06:17 2019

@author: greyoff
"""

import tensorflow as tf

class Network(object):
    def __init__(self, x, y, keep_prob, word_num = 91264, W_init = None, trainable = False):
        self.input_x = x
        self.input_y = y
        self.keep_prob = keep_prob
        self.W_init = W_init
        self.word_num = word_num
        self.trainable = trainable
        
        # 网络参数
        if W_init is not None:
            self.embedding_size = len(W_init[0])
            print('Using random initialized wv')
        else:
            self.embedding_size = 400 # 200 or 400
            print('use pretrained wv')
        self.label_num = self.input_y.get_shape()[-1]
        self.kernal_size = 5
        self.kernal_num = 256
        self.stride = 1
        
        # 训练参数
        self.learning_rate = 5e-4
        self.l2_lambda = 0.001
        
        self.logits = None
        self.loss = None
        self.predictions = None
        self.accuracy = None
        
        print('build model...')
        
        self.build_model()
        self.build_loss_accu()
        
        print('...build model successfully')
        
        self.train_op = None
        
        
    def build_model(self):
        
        
        with tf.variable_scope('embedding_layer'):
            if self.W_init is not None:
                W_init = tf.cast(self.W_init, tf.float32)  # 转换
                W = tf.Variable(W_init, name = 'embedding', trainable = self.trainable)
            else:
                W = tf.get_variable(name = 'embedding', shape = [self.word_num, self.embedding_size],
                                    initializer = tf.keras.initializers.he_uniform(), trainable = self.trainable)
            embedded_text = tf.nn.embedding_lookup(W, self.input_x)
            cnn_input = tf.expand_dims(embedded_text, -1, name = 'cnn_input')
        
        with tf.variable_scope('cnn_layer'):
            shape = [self.kernal_size, cnn_input.get_shape()[-2], 1, self.kernal_num]
            weights = tf.get_variable(name = 'cnn_weight', shape = shape,
                                      initializer = tf.keras.initializers.he_uniform())
            cnn_output = tf.nn.conv2d(cnn_input, weights, [1, self.stride, 1, 1], padding='VALID')
            
        with tf.variable_scope('max_pooling'):
            ksize = [1, cnn_output.get_shape()[1], 1, 1]
            max_pool = tf.nn.max_pool(cnn_output, ksize = ksize, strides = [1, 1, 1, 1], padding='VALID')
            flatten = tf.reshape(max_pool, [-1, max_pool.get_shape()[-1]])
            
        with tf.variable_scope('fc_with_dropout'):
            fc_dropout = tf.nn.dropout(flatten, keep_prob=self.keep_prob)
            
            W = tf.get_variable(name = 'fc_weight', shape = [fc_dropout.get_shape()[-1], self.label_num],
                                    initializer = tf.keras.initializers.he_uniform())
            b = tf.get_variable(name = 'fc_bias', shape = [self.label_num],
                                    initializer = tf.keras.initializers.he_uniform())
            out = tf.nn.xw_plus_b(fc_dropout, W, b)
            
            self.logits = tf.nn.relu(out)
        
        
        
    def build_loss_accu(self):
        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
            l2_loss=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.add(self.cross_entropy,tf.multiply(l2_loss,self.l2_lambda))
        with tf.name_scope('evaluaton'):
            self.predictions = tf.argmax(self.logits, 1, name="prediction")
            self.y_true = tf.arg_max(self.input_y, 1, name="y_true")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    def build_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        return self.train_op
            
if __name__ == '__main__':
    X = tf.placeholder(tf.int32, shape = [None, 100])
    Y = tf.placeholder(tf.int32, shape = [None, 9])
    keep_prob = tf.placeholder(tf.float32)
    
    model_test = Network(X,Y,keep_prob) 
    train_op = model_test.build_train_op()
        
        
        
            