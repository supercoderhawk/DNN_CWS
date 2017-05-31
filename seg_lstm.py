# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
import time

import constant
from transform_data_lstm import TransformDataLSTM
from seg_base import SegBase


class SegLSTM(SegBase):
  def __init__(self):
    SegBase.__init__(self)
    self.dtype = tf.float32
    # 参数初始化
    self.skip_window_left = constant.LSTM_SKIP_WINDOW_LEFT
    self.skip_window_right = constant.LSTM_SKIP_WINDOW_RIGHT
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.embed_size = 100
    self.hidden_units = 150
    self.tag_count = 4
    self.concat_embed_size = self.window_size * self.embed_size
    self.vocab_size = constant.VOCAB_SIZE
    self.alpha = 0.02
    self.lam = 0.001
    self.eta = 0.02
    self.dropout_rate = 0.2
    # 数据初始化
    trans = TransformDataLSTM()
    self.words_batch = trans.words_batch
    self.tags_batch = trans.labels_batch
    self.dictionary = trans.dictionary
    # 模型定义和初始化
    self.sess = tf.Session()
    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
    self.x = tf.placeholder(self.dtype, shape=[1, None, self.concat_embed_size])
    self.embeddings = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size], stddev=-1.0 / math.sqrt(self.embed_size),
                          dtype=self.dtype), dtype=self.dtype, name='embeddings')
    self.w = tf.Variable(
      tf.truncated_normal([self.tags_count, self.hidden_units], stddev=1.0 / math.sqrt(self.concat_embed_size)),
      dtype=self.dtype, name='w')
    self.b = tf.Variable(tf.zeros([self.tag_count, 1]), dtype=self.dtype, name='b')
    self.A = tf.Variable(tf.random_uniform([self.tag_count, self.tag_count], -0.05, 0.05), dtype=self.dtype, name='A')
    self.Ap = tf.placeholder(self.dtype, shape=self.A.get_shape())
    self.init_A = tf.Variable(tf.random_uniform([self.tag_count], -0.05, 0.05), dtype=self.dtype, name='init_A')
    self.init_Ap = tf.placeholder(self.dtype, shape=self.init_A.get_shape())
    self.update_A_op = (1 - self.lam) * self.A.assign_add(self.alpha * self.Ap)
    self.update_init_A_op = (1 - self.lam) * self.init_A.assign_add(self.alpha * self.init_Ap)
    self.sentence_holder = tf.placeholder(tf.int32, shape=[None, self.window_size])
    self.lookup_op = tf.nn.embedding_lookup(self.embeddings, self.sentence_holder)
    self.indices = tf.placeholder(tf.int32, shape=[None, 2])
    self.shape = tf.placeholder(tf.int32, shape=[2])
    self.values = tf.placeholder(self.dtype, shape=[None])
    self.map_matrix_op = tf.sparse_to_dense(self.indices, self.shape, self.values, validate_indices=False)
    self.map_matrix = tf.placeholder(self.dtype, shape=[self.tag_count, None])
    self.lstm = tf.contrib.rnn.LSTMCell(self.hidden_units)
    self.lstm_output, self.lstm_out_state = tf.nn.dynamic_rnn(self.lstm, self.x, dtype=self.dtype)
    tf.global_variables_initializer().run(session=self.sess)
    self.word_scores = tf.matmul(self.w, tf.transpose(self.lstm_output[0])) + self.b
    self.loss_scores = tf.multiply(self.map_matrix, self.word_scores)
    self.loss = tf.reduce_sum(self.loss_scores)
    self.lstm_variable = [v for v in tf.global_variables() if v.name.startswith('rnn')]
    self.params = [self.w, self.b] + self.lstm_variable
    self.regularization = list(map(lambda p: tf.assign_sub(p, self.lam * p), self.params))
    self.train = self.optimizer.minimize(self.loss, var_list=self.params)
    self.embedp = tf.placeholder(self.dtype, shape=[None, self.embed_size])
    self.embed_index = tf.placeholder(tf.int32, shape=[None])
    self.update_embed_op = tf.scatter_update(self.embeddings, self.embed_index, self.embedp)
    self.sentence_length = 1 #tf.placeholder(tf.int32, shape=[1])
    self.grad_embed = tf.gradients(tf.split(self.loss_scores, self.sentence_length),
                                   tf.split(self.x, self.sentence_length,1))
    self.saver = tf.train.Saver(self.params + [self.embeddings, self.A, self.init_A], max_to_keep=100)
    self.regu = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam),
                                                       self.params + [self.A, self.init_A])

  def model(self, embeds):
    scores = self.sess.run(self.word_scores, feed_dict={self.x: np.expand_dims(embeds, 0)})
    path = self.viterbi(scores, self.A.eval(self.sess), self.init_A.eval(self.sess))
    return path

  def train_exe(self):
    #self.sess.graph.finalize()
    last_time = time.time()
    for i in range(10):
      for sentence_index, (sentence, tags) in enumerate(zip(self.words_batch, self.tags_batch)):
        self.train_sentence(sentence, tags, len(tags))
        if sentence_index % 500 == 0:
          print(sentence_index)
          print(time.time() - last_time)
          last_time = time.time()
      print(self.sess.run(self.init_A))
      self.saver.save(self.sess, 'tmp/lstm-model%d.ckpt' % i)

  def train_sentence(self, sentence, tags, length):
    sentence_embeds = self.sess.run(self.lookup_op, feed_dict={self.sentence_holder: sentence}).reshape(
      [length, self.concat_embed_size])
    current_tags = self.model(sentence_embeds)
    diff_tags = np.subtract(tags, current_tags)
    update_index = np.where(diff_tags != 0)[0]
    update_length = len(update_index)

    if update_length == 0:
      return

    update_tags_pos = tags[update_index]
    update_tags_neg = current_tags[update_index]

    # update_embed = sentence_embeds[update_index]
    sparse_indices = np.stack(
      [np.concatenate([update_tags_pos, update_tags_neg], axis=-1), np.tile(update_index, [2])], axis=-1)

    sparse_values = np.concatenate([-1 * np.ones(update_length), np.ones(update_length)])
    output_shape = [self.tag_count, length]
    sentence_matrix = self.sess.run(self.map_matrix_op,
                                    feed_dict={self.indices: sparse_indices, self.shape: output_shape,
                                               self.values: sparse_values})
    #print(sentence_matrix)
    # 更新参数
    # self.sess.run(self.regu)
    self.sess.run(self.train,
                  feed_dict={self.x: np.expand_dims(sentence_embeds, 0), self.map_matrix: sentence_matrix})
    self.sess.run(self.regularization)

    # 更新词向量
    #sen_len = np.asarray(length,dtype=np.int32).reshape([1])
    self.sentence_length = length
    #print(tf.split(np.expand_dims(sentence_embeds,0), self.sentence_length,1))
    g = tf.gradients(tf.split(self.loss_scores, self.sentence_length),
                 tf.split(self.x, self.sentence_length, 1))
    grads = self.sess.run(self.grad_embed,
                          feed_dict={self.x: np.expand_dims(sentence_embeds, 0), self.map_matrix: sentence_matrix})

    print(grads.shape)
    '''
    embed_index = sentence[update_index]
    for i in range(update_length):
      embed = np.expand_dims(np.expand_dims(update_embed[:, i], 0), 0)
      grad = self.sess.run(self.grad_embed, feed_dict={self.x: embed,
                                                       self.map_matrix: np.expand_dims(sentence_matrix[:, i], 1)})[0]

      sentence_update_embed = (embed - self.alpha * grad) * (1 - self.lam)
      self.embeddings = self.sess.run(self.update_embed_op,
                                      feed_dict={
                                        self.embedp: sentence_update_embed.reshape([self.window_size, self.embed_size]),
                                        self.embed_index: embed_index[i, :]})
    '''
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

  def seg(self, sentence, model_path='tmp/lstm-model2.ckpt'):
    self.saver.restore(self.sess, model_path)
    seq = self.index2seq(self.sentence2index(sentence))
    sentence_embeds = tf.nn.embedding_lookup(self.embeddings, seq).eval(session=self.sess).reshape(
      [len(sentence), self.concat_embed_size])
    sentence_scores = self.sess.run(self.word_scores, feed_dict={self.x: np.expand_dims(sentence_embeds, 0)})
    init_A_val = self.init_A.eval(session=self.sess)
    A_val = self.A.eval(session=self.sess)
    # print(A_val)
    current_tags = self.viterbi(sentence_scores, A_val, init_A_val)
    return self.tags2words(sentence, current_tags), current_tags


if __name__ == '__main__':
  seg = SegLSTM()
  seg.train_exe()
  # print(seg.seg('我爱北京天安门'))
  # print(seg.seg('小明来自南京师范大学'))
  # print(seg.seg('小明是上海理工大学的学生'))
