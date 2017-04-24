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
    A_correct = np.array([[2, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [-1, 0, 0, 0]], dtype=np.int32)
    init_A_correct = np.array([1, -1, 0, 0])
    A_currect, init_A_currect, init_update = self.cws.gen_update_A(a, b)
    self.assertTrue(np.all(A_correct == A_currect))
    self.assertTrue(np.all(init_A_correct == init_A_currect))

  def test_viterbi(self):
    score = np.arange(10, 170, 10).reshape(4, 4).T
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    init_A = np.array([1, 1, 0, 0])
    current_path, current_score = self.cws.viterbi(score, A, init_A, True)
    correct_path = np.array([1, 3, 1, 3])
    correct_score = np.array([21, 102, 203, 364])
    self.assertTrue(np.all(current_path == correct_path))
    self.assertTrue(np.all(current_score == correct_score))

  def test_viterbi_all(self):
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    init_A = np.array([1, 1, 0, 0])
    score = np.arange(10, 170, 10).reshape(4, 4).T
    cur_path, cur_score = self.cws.viterbi_all(score, A, init_A, True)
    corr_path = np.array([3, 3, 3, 3])
    corr_score = np.array([40, 120, 240, 400])
    print(cur_score)
    self.assertTrue(np.all(cur_path == corr_path))
    self.assertTrue(np.all(cur_score == corr_score))
    # def test_sentence2index(self):
    #  pass

    # def test_index2seq(self):
    #  pass

    # def test_tags2words(self):
    #  pass


if __name__ == '__main__':
  main()
