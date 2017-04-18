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
    b = np.array([1, 1, 3, 1, 3], dtype=np.int32)
    A_correct = np.array([[2,0,0,0],[0,0,0,-1],[0,0,0,0],[0,-1,0,0]],dtype=np.int32)
    init_A_correct = np.array([1,-1,0,0])
    A_currect,init_A_currect,init_update = self.cws.gen_update_A(a, b)
    self.assertTrue(np.all(A_correct==A_currect))
    self.assertTrue(np.all(init_A_correct == init_A_currect))

  def test_viterbi(self):
    initial_prob = np.array([[0.6], [0.4]])
    trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    obs_prob = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    observations = np.array([0, 1, 1, 2, 1])

  def test_sentence2index(self):
    pass

  def test_index2seq(self):
    pass

  def test_tags2words(self):
    pass


if __name__ == '__main__':
  main()
