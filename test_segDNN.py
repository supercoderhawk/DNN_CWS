# -*- coding: UTF-8 -*-
from unittest import TestCase
from unittest import main
import numpy as np
from seg_dnn import SegDNN
import constant


class TestSegDNN(TestCase):
  def setUp(self):
    self.cws = SegDNN(constant.VOCAB_SIZE, 50, constant.DNN_SKIP_WINDOW)

  def test_gen_update_A(self):
    a = np.array([0, 0, 0, 1, 3], dtype=np.int32)
    b = np.array([1, 3, 0, 1, 3], dtype=np.int32)
    A_correct = np.array([[2,0,0,0],[0,0,0,-1],[0,0,0,0],[-1,0,0,0]],dtype=np.int32)
    init_A_correct = np.array([1,-1,0,0])
    A_currect,init_A_currect,init_update = self.cws.gen_update_A(a, b)
    self.assertTrue(np.all(A_correct==A_currect))
    self.assertTrue(np.all(init_A_correct == init_A_currect))

  def test_viterbi(self):
    score = np.arange(10,170,10).reshape(4,4).T
    A = np.array([[1,1,0,0],[0,0,1,1],[0,0,1,1],[1,1,0,0]])
    init_A = np.array([1,1,0,0])
    path = self.cws.viterbi(score,A,init_A)
    correct_path = np.array([1,3,1,3])
    self.assertTrue(np.all(path == correct_path))

  def test_sentence2index(self):
    pass

  def test_index2seq(self):
    pass

  def test_tags2words(self):
    pass


if __name__ == '__main__':
  main()
