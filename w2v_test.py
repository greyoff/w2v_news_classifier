#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:04:36 2019

@author: greyoff
"""

from w2v import model_load

model1_dir = './my_news_model_size400_win5'
model2_dir = './my_news_model_size200_win5'
model3_dir = './word2vec_779845.bin'

model1 = model_load(model1_dir)
model2 = model_load(model2_dir)
model3 = model_load(model3_dir)

print(model1.wv.index2word)
print(model2.wv.index2word)