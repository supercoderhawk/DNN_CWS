#-*- coding: UTF-8 -*-

from seg_dnn import SegDNN
import constant

def test_seg_dnn():
  cws = SegDNN(constant.VOCAB_SIZE,50,constant.DNN_SKIP_WINDOW)
  model = 'tmp/model2.ckpt'
  print(cws.seg('小明来自南京师范大学',model))
  print(cws.seg('小明是上海理工大学的学生',model))
  print(cws.seg('迈向充满希望的新世纪',model))
  print(cws.seg('2015世界旅游小姐大赛山东赛区冠军总决赛在威海举行',model))

if __name__ == '__main__':
  test_seg_dnn()