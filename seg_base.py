#-*- coding: UTF-8 -*-
import numpy as np


class SegBase:
  def __init__(self):
    self.TAGS = np.arange(4)
    self.TAG_MAPS = np.array([[0, 1], [2, 3], [2, 3], [0, 1]], dtype=np.int32)
    self.tags_count = len(self.TAG_MAPS)
    self.dictionary = {}
    self.skip_window_left = 0
    self.skip_window_right = 1

  def viterbi(self, emission, A, init_A, return_score=False):
    """
    维特比算法的实现，所有输入和返回参数均为numpy数组对象
    :param emission: 发射概率矩阵，对应于本模型中的分数矩阵，4*length
    :param A: 转移概率矩阵，4*4
    :param init_A: 初始转移概率矩阵，4
    :param return_score: 是否返回最优路径的分值，默认为False
    :return: 最优路径，若return_score为True，返回最优路径及其对应分值
    """

    length = emission.shape[1]
    path = np.ones([4, length], dtype=np.int32) * -1
    corr_path = np.zeros([length], dtype=np.int32)
    path_score = np.ones([4, length], dtype=np.float64) * (np.finfo('f').min / 2)
    path_score[:, 0] = init_A + emission[:, 0]

    for pos in range(1, length):
      for t in range(4):
        for prev in range(4):
          temp = path_score[prev][pos - 1] + A[prev][t] + emission[t][pos]
          if temp >= path_score[t][pos]:
            path[t][pos] = prev
            path_score[t][pos] = temp

    max_index = np.argmax(path_score[:, -1])
    corr_path[length - 1] = max_index
    for i in range(length - 1, 0, -1):
      max_index = path[max_index][i]
      corr_path[i - 1] = max_index
    if return_score:
      return corr_path, path_score[max_index, :]
    else:
      return corr_path

  def sentence2index(self, sentence):
    index = []
    for word in sentence:
      if word not in self.dictionary:
        index.append(0)
      else:
        index.append(self.dictionary[word])

    return index

  def index2seq(self, indices):
    ext_indices = [1] * self.skip_window_left
    ext_indices.extend(indices + [2] * self.skip_window_right)
    seq = []
    for index in range(self.skip_window_left, len(ext_indices) - self.skip_window_right):
      seq.append(ext_indices[index - self.skip_window_left: index + self.skip_window_right + 1])

    return seq

  def tags2words(self, sentence, tags):
    words = []
    word = ''
    for tag_index, tag in enumerate(tags):
      if tag == 0:
        words.append(sentence[tag_index])
      elif tag == 1:
        word = sentence[tag_index]
      elif tag == 2:
        word += sentence[tag_index]
      else:
        words.append(word + sentence[tag_index])
        word = ''
    # 处理最后一个标记为I的情况
    if word != '':
      words.append(word)

    return words