# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
import time

import constant
from transform_data_lstm import TransformDataLSTM


class SegLSTM:
  def __init__(self):
    self.dtype = tf.float32
    self.skip_window_left = constant.LSTM_SKIP_WINDOW_LEFT
    self.skip_window_right = constant.LSTM_SKIP_WINDOW_RIGHT
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.embed_size = 50
    self.hidden_units = 100
    self.tag_count = 4
    self.concat_embed_size = self.window_size * self.embed_size
    trans = TransformDataLSTM()
    self.words_batch = trans.words_batch
    self.tags_batch = trans.labels_batch
    self.vocab_size = constant.VOCAB_SIZE
    self.alpha = 0.02
    self.lam = 0.001
    self.sess = tf.Session()
    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
    self.x = tf.placeholder(self.dtype, shape=[self.concat_embed_size, None])
    self.x_plus = tf.placeholder(self.dtype, shape=[1, None, self.concat_embed_size])
    self.embeddings = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embed_size], -1.0 / math.sqrt(self.embed_size),
                        1.0 / math.sqrt(self.embed_size),
                        dtype=self.dtype), dtype=self.dtype, name='embeddings')
    self.w = tf.Variable(tf.zeros([self.tag_count, self.hidden_units]), dtype=self.dtype)
    self.b = tf.Variable(tf.zeros([self.tag_count, 1]), dtype=self.dtype)
    self.A = tf.Variable(tf.zeros([self.tag_count, self.tag_count]), dtype=self.dtype)
    self.Ap = tf.placeholder(self.dtype, shape=self.A.get_shape())
    self.init_A = tf.Variable(tf.zeros([self.tag_count]), dtype=self.dtype)
    self.init_Ap = tf.placeholder(self.dtype, shape=self.init_A.get_shape())
    self.update_A_op = (1 - self.lam) * self.A.assign_add(self.alpha * self.Ap)
    self.update_init_A_op = (1 - self.lam) * self.init_A.assign_add(self.alpha * self.init_Ap)
    self.sentence_holder = tf.placeholder(tf.int32, shape=[None, self.window_size])
    self.lookup_op = tf.nn.embedding_lookup(self.embeddings, self.sentence_holder)
    self.indices = tf.placeholder(tf.int32, shape=[None, 2])
    self.shape = tf.placeholder(tf.int32, shape=[2])
    self.values = tf.placeholder(self.dtype, shape=[None])
    self.map_matrix_op = tf.sparse_to_dense(self.indices, self.shape, self.values, validate_indices=False)
    self.map_matrix = tf.placeholder(self.dtype, shape=[self.tag_count, None], name='mm')
    self.lstm = tf.contrib.rnn.LSTMCell(self.hidden_units)
    self.lstm_output, self.lstm_out_state = tf.nn.dynamic_rnn(self.lstm, self.x_plus, dtype=self.dtype)
    tf.global_variables_initializer().run(session=self.sess)
    self.word_scores = tf.matmul(self.w, tf.transpose(self.lstm_output[0])) + self.b
    self.loss = tf.reduce_sum(tf.multiply(self.map_matrix, self.word_scores))
    self.lstm_variable = [v for v in tf.global_variables() if v.name.startswith('rnn')]
    self.params = [self.w, self.b]
    self.params.extend(self.lstm_variable)
    self.regularization = list(map(lambda p: tf.assign_sub(p, self.lam * p), self.params))
    self.train = self.optimizer.minimize(self.loss, var_list=self.params)
    self.embedp = tf.placeholder(self.dtype, shape=[None, self.embed_size])
    self.embed_index = tf.placeholder(tf.int32, shape=[None])
    self.update_embed_op = tf.scatter_update(self.embeddings, self.embed_index, self.embedp)
    self.grad_embed = tf.gradients(tf.multiply(self.map_matrix, self.word_scores), self.x_plus)

  def model(self, embeds):
    scores = self.sess.run(self.word_scores, feed_dict={self.x_plus: np.expand_dims(embeds.T, 0)})
    path = self.viterbi(scores, self.A.eval(self.sess), self.init_A.eval(self.sess))
    return path

  def train_exe(self):
    self.sess.graph.finalize()
    last_time = time.time()
    saver = tf.train.Saver([self.embeddings, self.A,self.init_A].extend(self.params), max_to_keep=100)
    for sentence_index, (sentence, tags) in enumerate(zip(self.words_batch, self.tags_batch)):
      self.train_sentence(sentence, tags, len(tags))
      if sentence_index % 500 == 0:
        print(time.time() - last_time)
        last_time = time.time()


  def train_sentence(self, sentence, tags, length):
    sentence_embeds = self.sess.run(self.lookup_op, feed_dict={self.sentence_holder: sentence}).reshape(
      [length, self.concat_embed_size]).T
    current_tags = self.model(sentence_embeds)
    diff_tags = np.subtract(tags, current_tags)
    update_index = np.where(diff_tags != 0)[0]
    update_length = len(update_index)

    if update_length == 0:
      return

    update_tags_pos = tags[update_index]
    update_tags_neg = current_tags[update_index]

    update_embed = sentence_embeds[:, update_index]
    sparse_indices = np.stack(
      [np.concatenate([update_tags_pos, update_tags_neg], axis=-1), np.tile(np.arange(update_length), [2])], axis=-1)

    sparse_values = np.concatenate([-1 * np.ones(update_length), np.ones(update_length)])
    output_shape = [self.tag_count, update_length]
    sentence_matrix = self.sess.run(self.map_matrix_op,
                                    feed_dict={self.indices: sparse_indices, self.shape: output_shape,
                                               self.values: sparse_values})
    # 更新参数
    self.sess.run(self.train,
                  feed_dict={self.x_plus: np.expand_dims(update_embed.T, 0), self.map_matrix: sentence_matrix})
    self.sess.run(self.regularization)

    # 更新词向量
    embed_index = sentence[update_index]
    for i in range(update_length):
      embed = np.expand_dims(np.expand_dims(update_embed[:, i], 0),0)
      grad = self.sess.run(self.grad_embed, feed_dict={self.x_plus: embed,
                                                       self.map_matrix: np.expand_dims(sentence_matrix[:, i], 1)})[0]

      sentence_update_embed = (embed + self.alpha * grad) * (1 - self.lam)
      self.embeddings = self.sess.run(self.update_embed_op,
                                      feed_dict={
                                        self.embedp: sentence_update_embed.reshape([self.window_size, self.embed_size]),
                                        self.embed_index: embed_index[i, :]})

    # 更新转移矩阵
    A_update, init_A_update, update_init = self.gen_update_A(tags, current_tags)
    if update_init:
      self.sess.run(self.update_init_A_op, feed_dict={self.init_Ap: init_A_update})
    self.sess.run(self.update_A_op, {self.Ap: A_update})

  @staticmethod
  def gen_update_A(correct_tags, current_tags):
    A_update = np.zeros([4, 4], dtype=np.float32)
    init_A_update = np.zeros([4], dtype=np.float32)
    before_corr = correct_tags[0]
    before_curr = current_tags[0]
    update_init = False

    if before_corr != before_curr:
      init_A_update[before_corr] += 1
      init_A_update[before_curr] -= 1
      update_init = True

    for _, (corr_tag, curr_tag) in enumerate(zip(correct_tags[1:], current_tags[1:])):
      if corr_tag != curr_tag or before_corr != before_curr:
        A_update[before_corr, corr_tag] += 1
        A_update[before_curr, curr_tag] -= 1
      before_corr = corr_tag
      before_curr = curr_tag

    return A_update, init_A_update, update_init

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


if __name__ == '__main__':
  seg = SegLSTM()
  seg.train_exe()
