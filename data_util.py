#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:28:01 2019

@author: greyoff
"""
import os
import numpy as np
import jieba
from sklearn.model_selection import train_test_split
from w2v import model_load as load_w2v


class word_vecter(object):
    """
    build dictionary
    """
    def __init__(self,model_dir):
        self.model_dir = model_dir
        self.model = load_w2v(model_dir)
        self.wv = self.model.wv
        self.index2word = ['<空>']+self.model.wv.index2word  # list 加入了空白词
        self.word2index = dict(zip(self.index2word, list(range(len(self.index2word))))) # dict

class Data(object):
    """
    build data
    """
    def __init__(self, data_dir, model_dir, max_length = 100):
        
        print('start loading!')
        self.data_dir = data_dir
        self.w2v_model = word_vecter(model_dir)
        self.max_length = max_length
        
        print('build [index, word, vector] relationship')
        self.index2word = self.w2v_model.index2word
        self.word2index = self.w2v_model.word2index
        self.index2vec = self._build_dict() # array
        
        print('load texts and build [label, index] relationship')
        self.all_text = None
        self.index2label = None
        self.label2index = None
        self.label_num = None
        self._load_all_text()
        
        
        print('build input X and Y matrice')
        self.X = None
        self.Y = None
        self._build_x_y()  # y is one-hot
        
        print('load data successfully')
        
        
    def _build_dict(self):
        W = []
        vec_len = len(self.w2v_model.wv['我'])
        W.append([0]*vec_len)
        for word in self.index2word[1:]:
            W.append(self.w2v_model.wv[word])
        return np.array(W)
        
    def _load_all_text(self):
        self.index2label = []
        for label_name in os.listdir(self.data_dir):
            if '._' not in label_name:                           # 傻逼MAC气死我了
                self.index2label.append(label_name)  # list
        self.label_num = len(self.index2label)
        self.label2index = dict(zip(self.index2label,list(range(self.label_num)))) # dict
        
        self.all_text = []
        for label in self.index2label:
            text_in_label = []
            for txt in os.listdir(self.data_dir+'/'+label):
                if '._' not in txt:
                    if os.path.splitext(txt)[1] == '.txt':
                        file_dir = self.data_dir+'/'+label+'/'+txt
                        with open(file_dir, encoding = 'gb18030') as f:
                            print(file_dir)
                            document = f.read()
                            text_in_label.append(document)
            self.all_text.append(text_in_label)
    
    def _build_x_y(self):
        X = []
        Y = []
        for i, texts in enumerate(self.all_text):
            L = [0] * self.label_num   # one-hot
            L[i] = 1
            Y.extend([L]*len(texts))
            
            S = []
            for text in texts:
                s = [0] * self.max_length
                for i, word in enumerate(jieba.cut(text)):
                    if i<self.max_length:
                        try:
                            s[i] = self.word2index[word]
                        except:
                            pass
                S.append(s)
            
            X.extend(S)
        
        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def get_splited_train_test(self, test_size = 0.1):
        x_train, x_dev, y_train, y_dev = train_test_split(self.X, self.Y, test_size=test_size)
        return x_train, x_dev, y_train, y_dev
    
def get_next_batch(i,x_train, y_train, batch_size, batch_num):
    if i == batch_num-1:
        x_batch = x_train[i*batch_size:]
        y_batch = y_train[i*batch_size:]
    else:
        x_batch = x_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]
    return x_batch, y_batch

if __name__ == '__main__':
    data_dir = './Reduced'
    model_dir = './my_news_model_size200_win5'
    news_data = Data(data_dir, model_dir)
    