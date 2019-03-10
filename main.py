#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:44:04 2019

@author: greyoff
"""

import tensorflow as tf
from data_util import Data
from data_util import get_next_batch
from network import Network

# 数据模型等目录
model_info = 'news_classifier_w2v_random_ft'  # 1.w2v_random 2.w2v_400 3.w2v_big (and with fine-tuning) 词向量长度均为400
data_dir = './Reduced'
w2v_model_dir = './my_news_model_size400_win5'  # 1.随机初始化：False 2.用70M新闻语料跑出来的词向量：./my_news_model_size400_win5
use_w2v = False                                 # 3.别人用1.5G新闻跑出来的词向量： ./word2vec_779845.bin
log_dir =  './logs' + '/' + model_info
save_dir = './train_dir' + '/' + model_info

# 训练用参数
keep_prob_init = 0.5
batch_size = 128
epoch_num = 200
w2v_trainable = True  # 是否对词向量进行微调（fine-tuning）

# 保存模型时的参考准确率
ref_accu = 0.85

print('Load data...')
news_data = Data(data_dir, w2v_model_dir)
x_train, x_dev, y_train, y_dev = news_data.get_splited_train_test()

train_num = len(y_train)
test_num = len(y_dev)
print('train:{}, test:{}'.format(train_num,test_num))
print('...successfully')

with tf.Graph().as_default():
    print('Build model and train_op...')
    X = tf.placeholder(tf.int32, shape = [None, 100])
    Y = tf.placeholder(tf.int32, shape = [None, 9])
    keep_prob = tf.placeholder(tf.float32)
    
    if use_w2v:
        W_init = news_data.index2vec
    else:
        W_init = None
    model = Network(X, Y, keep_prob, W_init = W_init, trainable=w2v_trainable) # or None
    train_op = model.build_train_op()
    print('...successfully')
    
    print('Build summary...')
    summary_train_1 = tf.summary.scalar('train_loss', model.loss)
    summary_train_2 = tf.summary.scalar('train_accuracy', model.accuracy)
    summary_test_1 = tf.summary.scalar('test_loss', model.loss)
    summary_test_2 = tf.summary.scalar('test_accuracy', model.accuracy)
    summary_train = tf.summary.merge([summary_train_1, summary_train_2], name = 'summary_train')
    summary_test = tf.summary.merge([summary_test_1, summary_test_2], name = 'summary_test')
    print('...successfully')
    
    with tf.Session() as sess:
        step = 0
        max_accu = 0
        num_model_saved = 0
        batch_num = int((train_num-1)/batch_size)+1
        
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        test_feed = {X: x_dev, Y: y_dev, keep_prob: 1.0}
        
        print('Begin training...')
        for epoch in range(epoch_num):
            for i in range(batch_num):
                step += 1
                x_batch, y_batch = get_next_batch(i,x_train, y_train, batch_size, batch_num)
                _, loss, accuracy  = sess.run([train_op, model.loss, model.accuracy], 
                                              feed_dict = {X: x_batch, Y: y_batch, keep_prob: keep_prob_init})
                print('the current step is {}'.format(step))
                print('the currrent epoch is {}, batch is {}!\ntrain loss: {} train accuracy: {}'.format(epoch, i, loss, accuracy))
            
            print('************ the {} epoch is over! **********'.format(epoch))
            train_summary_in_epoch, train_loss, train_accuracy = sess.run([summary_train, model.loss, model.accuracy],
                                                                          feed_dict = {X :x_train, Y: y_train, keep_prob: 1.0})
            summary_writer.add_summary(train_summary_in_epoch, epoch)
            test_summary_in_epoch, test_loss, test_accuracy = sess.run([summary_test, model.loss, model.accuracy], feed_dict = test_feed)
            summary_writer.add_summary(test_summary_in_epoch, epoch)
            print('train loss: {} train accuracy: {}'.format(train_loss, train_accuracy))
            print('test loss: {}, test accuracy: {}'.format(test_loss, test_accuracy))
            
            max_accu = max(max_accu, test_accuracy)
            
            # 保存模型
            if epoch>10 and test_accuracy>ref_accu:
                path_model = save_dir + '/' + 'epoch_{}'.format(epoch) + '_accuracy_{:.5f}'.format(test_accuracy)
                builder = tf.saved_model.builder.SavedModelBuilder(path_model)
                print('Saved model checkpoint to:', path_model)
                
                builder.add_meta_graph_and_variables(sess, [model_info])
                builder.save()
                
                num_model_saved += 1 
                print('The model has been saved!')
                print('Now we have trained {} epoches! It\'s the {} model we have saved!'.format(epoch, num_model_saved))
                
            if num_model_saved>0 and num_model_saved%5==0:
                    ref_accu = max_accu
            
            print('*********************************************')
        
        if num_model_saved==0:
                print('The {} epoches training process is over but no model meets our criterion!'.format(epoch_num))
                print('Then the last model will be saved!')
                #timestring = str(int(time.time()))
                path_model = save_dir + '/' + 'lastmodel' +'_accuracy_{:.5f}'.format(test_accuracy)
                builder = tf.saved_model.builder.SavedModelBuilder(path_model)
                print('Saved model checkpoint to:', path_model)
                
                builder.add_meta_graph_and_variables(sess, [model_info])
                builder.save()
        
        print('...end')
        print('DONE')
    
    

