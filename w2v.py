#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:15:46 2019

@author: greyoff
"""
from gensim.models import word2vec
import gensim
import os

def model_train(train_text_dir, size = 200, window = 5):
    model_info = 'my_news_model_size{}_win{}'.format(size,window)
    sentences =word2vec.Text8Corpus(train_text_dir)
    print('begin training')
    model = word2vec.Word2Vec(sentences, sg=1, size=size,  window=window,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
    print('begin saving')
    model.save('./'+model_info)
    model.wv.save_word2vec_format('./'+model_info+'.bin.gz',binary = True)
    print('save model successfully!')
    return model

def model_load(model_dir):
    model_type = os.path.splitext(model_dir)[-1]
    if model_type == '.gz' or model_type == '.bin':
        model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=True)
    else:
        model = gensim.models.Word2Vec.load(model_dir)
    return model

if __name__=='__main__':
    print (os.path.abspath('.'))
    train_dir = u"all_text_cut.txt"
    #model1 = model_train(train_dir) # 200
    model2 = model_train(train_dir, size = 400) #400