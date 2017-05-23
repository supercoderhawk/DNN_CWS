# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math

class SegLSTM:
  def __init__(self):
    self.dtype = tf.float64
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.embed_size = 50
    self.hidden_units = 100
    self.tag_count = 4
    self.concat_embed_size = self.window_size * self.embed_size
    self.words_batch = None
    self.tags_batch = None
    self.vocab_size = 4000
    self.sess = tf.Session()
    self.x = tf.placeholder(tf.int32, shape=[self.concat_embed_size, None])
    self.embeddings = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embed_size], -1.0 / math.sqrt(self.embed_size),
                        1.0 / math.sqrt(self.embed_size),
                        dtype=tf.float64), dtype=tf.float64, name='embeddings')
    self.w = tf.Variable(tf.zeros([self.tag_count,self.hidden_units]),dtype=tf.float32)
    self.b = tf.Variable(tf.zeros([self.tag_count]),dtype=tf.float32)
    self.lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_units,reuse=True)
    self.A = tf.Variable(tf.zeros([self.tag_count,self.tag_count]),dtype=tf.float32)
    self.Ap = tf.placeholder(tf.float32,shape=self.A.get_shape())
    self.init_A = tf.Variable(tf.zeros([self.tag_count]),dtype=tf.float32)
    self.init_Ap = tf.placeholder(tf.float32,shape=self.init_A.get_shape())
    self.sentence_holder = tf.placeholder(tf.int32, shape=[None, self.window_size])
    self.lookup_op = tf.nn.embedding_lookup(self.embeddings, self.sentence_holder)

  def model(self,input):
    output, out_state = tf.nn.dynamic_rnn(self.lstm, input, dtype=tf.float32)
    with tf.variable_scope("rnn"):
      tf.initialize_local_variables()
      path = self.viterbi(output,self.A.eval(),self.init_A.eval())
    return path

  def train(self):
    pass

  def train_sentence(self,sentence,tags,length):
    sentence_embeds = self.sess.run(self.lookup_op,feed_dict={self.sentence_holder:sentence}).reshape(
      [length, self.concat_embed_size])
    current_tags = self.model(sentence_embeds)




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