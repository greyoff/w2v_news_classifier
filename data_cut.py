#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:42:06 2019

@author: greyoff
"""

#获取默认路径
import os
import jieba
#from gensim.models import Word2Vec

def get_subfile(file_dir):
    all_file = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                all_file.append(os.path.join(root,file))
    return all_file

def get_text_cut(files, outfile = './all_text_cut.txt'):
    for file in files:
        with open(file, encoding = 'gb18030') as f:
            print(file,'...begining')
            document = f.read()
            documemt_cut = jieba.cut(document)
            result = ' '.join(documemt_cut)
            #result = result.encode('utf-8') # error
            with open(outfile, 'a+', encoding = 'utf-8') as f2: # 'a+' ==a+r（可追加可写，文件若不存在就创建）
                f2.write('\n')
                f2.write(result)
                print(file,'write successfully')
    
if __name__=='__main__':
    print (os.path.abspath('.'))
    txt_dir = './Reduced'
    all_file_names = get_subfile(txt_dir)
    get_text_cut(all_file_names)
