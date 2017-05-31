# -*- coding: UTF-8 -*-
import numpy as np
#import jieba
from seg_dnn import SegDNN
from seg_lstm import SegLSTM
from utils import estimate_cws
import constant


def test_seg_dnn():
  '''
  cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
  sentence = '迈向充满希望的新世纪'
  # model = 'tmp/4.29-w3-normal/model15.ckpt'
  model = 'tmp/4.29-100/model99.ckpt'
  model = 'tmp/model0.ckpt'
  print(cws.seg('小明来自南京师范大学', model))
  print(cws.seg('小明是上海理工大学的学生', model))
  print(cws.seg('小明是清华大学的学生', model))
  print(cws.seg('我爱北京天安门', model))
  print(cws.seg('上海理工大学', model))
  print(cws.seg('上海海洋大学'))
  print(cws.seg(sentence, model))
  #print('/'.join(jieba.cut('小明是上海理工大学的学生')))
  seq = cws.index2seq(cws.sentence2index(sentence))
  seq = np.array(seq, dtype=np.int32).flatten()
  '''
  seg = SegLSTM()
  # seg.train_exe()
  #print(seg.seg('我爱北京天安门'))
  #print(seg.seg('小明来自南京师范大学'))
  #print(seg.seg('小明是上海理工大学的学生'))
  print(seg.seg('小明来自南京师范大学'))
  test(seg,'tmp/lstm-model1.ckpt')
  # print(seq)
  # cal_val(seq)
  # print(cws.seg('2015世界旅游小姐大赛山东赛区冠军总决赛在威海举行',model))


def cal_val(seq):
  embeddings = np.load('data/dnn/embeddings.npy')
  w2 = np.load('data/dnn/w2.npy')
  w3 = np.load('data/dnn/w3.npy')
  b2 = np.load('data/dnn/b2.npy')
  b3 = np.load('data/dnn/b3.npy')
  # b2 = np.expand_dims(b2.flatten(),0)
  # A = np.load('data/dnn/A.npy')
  # init_A = np.load('data/dnn/init_A.npy')
  # print(w2)
  # x = np.reshape(embeddings[seq],[10,150])
  # print(x[0])
  # b2 = np.tile(b2,[10])
  # print(np.matmul(w2,x.T))
  # print(b2.shape)
  # val = np.matmul(w3,sigmoid(np.add(np.matmul(w2,x.T),b2)))+b3
  # val = w3*sigmoid(w2*x.T+b2)+b3
  # print(val.T)


def test(cws, model):
  with open('data/test.utf8', 'r', encoding='utf-8') as file:
    lines = file.read().splitlines()
    sentences = lines[::2][:100]
    labels = lines[1::2][:100]
    corr_count = 0
    re_count = 0
    total_count = 0
    for _, (sentence, label) in enumerate(zip(sentences, labels)):
      label = np.array(list(map(lambda s: int(s), label.split(' '))))
      _, tag = cws.seg(sentence, model)
      cor_count, prec_count, recall_count = estimate_cws(tag, np.array(label))
      corr_count += cor_count
      re_count += recall_count
      total_count += prec_count
      # if(corr_count != prec_count):
      #  print(cws.tags2words(sentence,tag))

      # diff = np.subtract(tag,np.array(label))
      # if sum()
      # print(np.where(diff == 0))
      # corr_count += len(np.where(diff == 0)[0])
      # total_count += len(label)
    prec = corr_count / total_count
    recall = corr_count / re_count
    print(prec)
    print(recall)
    print(2 * prec * recall / (prec + recall))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
  test_seg_dnn()
