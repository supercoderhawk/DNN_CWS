# -*- coding: UTF-8 -*-
import numpy as np
from seg_dnn import SegDNN
from seg_lstm import SegLSTM
from utils import estimate_cws
import constant


def test_seg_dnn():
  cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)
  model = 'tmp/model0.ckpt'
  print(cws.seg('小明来自南京师范大学', model))
  print(cws.seg('小明是上海理工大学的学生', model))
  print(cws.seg('小明是清华大学的学生', model))
  print(cws.seg('我爱北京天安门', model))
  print(cws.seg('上海理工大学', model))
  print(cws.seg('上海海洋大学'))
  print(cws.seg('迈向充满希望的新世纪', model))


def test_seg_lstm():
  seg = SegLSTM()
  model = 'tmp/lstm-model1.ckpt'
  print(seg.seg('小明来自南京师范大学', model, debug=True))
  print(seg.seg('小明是上海理工大学的学生', model))
  print(seg.seg('迈向充满希望的新世纪', model))
  print(seg.seg('2015世界旅游小姐大赛山东赛区冠军总决赛在威海举行', model))
  test(seg, model)


def test(cws, model):
  with open('tmp/test.utf8', 'r', encoding='utf-8') as file:
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
    prec = corr_count / total_count
    recall = corr_count / re_count

    print(prec)
    print(recall)
    print(2 * prec * recall / (prec + recall))


if __name__ == '__main__':
  # test_seg_dnn()
  test_seg_lstm()
