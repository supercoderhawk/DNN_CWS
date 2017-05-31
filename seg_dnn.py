# -*- coding: UTF-8 -*-
import tensorflow as tf
import math
import numpy as np
import time
from transform_data_dnn import TransformDataDNN
import constant


class SegDNN:
  def __init__(self, vocab_size, embed_size, skip_window):
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.skip_window = skip_window
    self.alpha = 0.02
    self.h = 300
    self.tags = [0, 1, 2, 3]
    self.tags_count = len(self.tags)
    self.window_length = 2 * self.skip_window + 1
    self.concat_embed_size = self.embed_size * self.window_length
    trans_dnn = TransformDataDNN(self.skip_window)
    self.dictionary = trans_dnn.dictionary
    self.words_batch = trans_dnn.words_batch
    self.tags_batch = trans_dnn.labels_batch
    self.words_count = trans_dnn.words_count
    self.sess = None
    self.x = tf.placeholder(tf.float64, shape=[self.concat_embed_size, None], name='x')
    self.map_matrix = tf.placeholder(tf.float64, shape=[4, None], name='mm')
    self.embeddings = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embed_size], -1.0 / math.sqrt(self.embed_size),
                        1.0 / math.sqrt(self.embed_size),
                        dtype=tf.float64), dtype=tf.float64, name='embeddings')
    self.w2 = tf.Variable(
      tf.random_uniform([self.h, self.concat_embed_size], -4.0 / math.sqrt(self.concat_embed_size),
                        4 / math.sqrt(self.concat_embed_size),
                        dtype=tf.float64), dtype=tf.float64, name='w2')
    self.b2 = tf.Variable(tf.zeros([self.h, 1], dtype=tf.float64), dtype=tf.float64, name='b2')

    self.w3 = tf.Variable(
      tf.random_uniform([self.tags_count, self.h], -4.0 / math.sqrt(self.h), 4.0 / math.sqrt(self.h), dtype=tf.float64),
      dtype=tf.float64, name='w3')
    self.b3 = tf.Variable(tf.zeros([self.tags_count, 1], dtype=tf.float64), dtype=tf.float64, name='b3')
    self.word_score = tf.add(tf.matmul(self.w3, tf.sigmoid(tf.add(tf.matmul(self.w2, self.x), self.b2))), self.b3)
    self.params = [self.w2, self.b2, self.w3, self.b3]
    self.A = tf.Variable(tf.random_uniform([4, 4], -1, 1, dtype=tf.float64), dtype=tf.float64, name='A')
    self.init_A = tf.Variable(tf.random_uniform([4], -1, 1, dtype=tf.float64), dtype=tf.float64, name='init_A')
    self.Ap = tf.placeholder(tf.float64, shape=self.A.get_shape())
    self.init_Ap = tf.placeholder(tf.float64, shape=self.init_A.get_shape())
    self.embedp = tf.placeholder(tf.float64, shape=[None, self.embed_size])
    self.embed_index = tf.placeholder(tf.int32, shape=[None])
    self.update_embed_op = tf.scatter_update(self.embeddings, self.embed_index, self.embedp)
    self.lam = 0.0001
    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
    self.update_A_op = (1 - self.lam) * self.A.assign_add(self.alpha * self.Ap)
    self.update_init_A_op = (1 - self.lam) * self.init_A.assign_add(self.alpha * self.init_Ap)
    self.loss = -tf.reduce_sum(tf.multiply(self.map_matrix, self.word_score))
    self.grad_embed = tf.gradients(tf.multiply(self.map_matrix, self.word_score), self.x)
    self.update_embed = self.alpha * (self.grad_embed[0]) + (1 - self.lam) * self.x
    self.train_loss = self.optimizer.minimize(self.loss, var_list=self.params)
    self.indices = tf.placeholder(tf.int32, shape=[None, 2])
    self.shape = tf.placeholder(tf.int32, shape=[2])
    self.values = tf.placeholder(tf.float64, shape=[None])
    self.gen_map = tf.sparse_to_dense(self.indices, self.shape, self.values, validate_indices=False)
    self.sentence_holder = tf.placeholder(tf.int32, shape=[None, self.window_length])
    self.lookup_op = tf.nn.embedding_lookup(self.embeddings, self.sentence_holder)
    self.params_regularization = list(map(lambda p: tf.assign_sub(p, self.lam * p), self.params))
    self.line_index = np.arange(4, dtype=np.int32)
    self.sentence_index = 0

  def train(self):
    """
    用于训练模型
    :param vocab_size:
    :param embed_size:
    :param skip_window:
    :return:
    """

    print('start...')

    self.sess = tf.Session()
    params = [self.embeddings, self.A, self.init_A].extend(self.params)
    saver = tf.train.Saver(params, max_to_keep=100)
    train_writer = tf.summary.FileWriter('logs', self.sess.graph)
    init = tf.global_variables_initializer()
    init.run(session=self.sess)
    self.sess.graph.finalize()
    loss = []
    count = 10

    for i in range(count):
      loss.append(self.train_exe() / 10000)
      print(i)
      saver.save(self.sess, 'tmp/model%d.ckpt' % i)
      train_writer.flush()
    print(loss)
    train_writer.flush()
    self.sess.close()

  def train_exe(self):
    """
    进行一轮训练
    :return: 
    """
    start = time.time()
    time_all = 0.0
    start_c = 0
    for sentence_index, (sentence, tags) in enumerate(zip(self.words_batch, self.tags_batch)):
      self.sentence_index = sentence_index
      start_s = time.time()
      self.train_sentence(sentence, tags, len(tags))
      start_c += time.time() - start_s
      time_all += time.time() - start_s

      if sentence_index % 2000 == 0:
        print('s:' + str(sentence_index))
        print(start_c)
        print(time_all / 60)
        start_c = 0

    loss = 0.0
    # for sentence_index, (sentence, tags) in enumerate(zip(self.words_batch, self.tags_batch)):
    #  loss += self.cal_sentence_loss(sentence, tags, len(tags))
    # print(loss)
    print(time.time() - start)
    return math.fabs(loss)

  def train_sentence(self, sentence, tags, length):
    """
    对每个句子进行训练
    :param sentence: 
    :param tags: 
    :param length: 
    :return: 
    """
    sentence_embeds = self.sess.run(self.lookup_op, feed_dict={self.sentence_holder: sentence}).reshape(
      [length, self.concat_embed_size]).T
    sentence_scores = self.sess.run(self.word_score, feed_dict={self.x: sentence_embeds})

    init_A_val = self.init_A.eval(session=self.sess)
    A_val = self.A.eval(session=self.sess)
    current_tags = self.viterbi(sentence_scores, A_val, init_A_val)  # 当前参数下的最优路径
    diff_tags = np.subtract(tags, current_tags)
    update_index = np.where(diff_tags != 0)[0]  # 标签不同的字符位置
    update_length = len(update_index)

    # 完全正确
    if update_length == 0:
      return 0, 0

    update_pos_tags = tags[update_index]  # 需要更新的字符的位置对应的正确字符标签
    update_neg_tags = current_tags[update_index]  # 需要更新的字符的位置对应的错误字符标签
    update_embed = sentence_embeds[:, update_index]
    sparse_indices = np.stack(
      [np.concatenate([update_pos_tags, update_neg_tags], axis=-1), np.tile(np.arange(update_length), [2])], axis=-1)

    sparse_values = np.concatenate([np.ones(update_length), -1 * np.ones(update_length)])
    output_shape = [4, update_length]
    sentence_matrix = self.sess.run(self.gen_map, feed_dict={self.indices: sparse_indices, self.shape: output_shape,
                                                             self.values: sparse_values})
    self.update_params(sentence_matrix, update_embed, sentence[update_index], update_length)
    # 更新转移矩阵
    A_update, init_A_update, update_init = self.gen_update_A(tags, current_tags)
    if update_init:
      self.sess.run(self.update_init_A_op, feed_dict={self.init_Ap: init_A_update})

    self.sess.run(self.update_A_op, {self.Ap: A_update})

  def update_params(self, sen_matrix, embeds, embed_index, update_length):
    """
    
    :param sen_matrix: 4*length
    :param embeds: 150*length
    :param embed_index: length*3
    :param update_length: 
    :return: 
    """

    self.sess.run(self.train_loss, feed_dict={self.x: embeds, self.map_matrix: sen_matrix})
    self.sess.run(self.params_regularization)

    for i in range(update_length):
      embed = np.expand_dims(embeds[:, i], 1)
      grad = self.sess.run(self.grad_embed, feed_dict={self.x: embed,
                                                       self.map_matrix: np.expand_dims(sen_matrix[:, i], 1)})[0]
      update_embed = (embed + self.alpha * grad) * (1 - self.lam)
      self.embeddings = self.sess.run(self.update_embed_op,
                                      feed_dict={
                                        self.embedp: update_embed.reshape([self.window_length, self.embed_size]),
                                        self.embed_index: embed_index[i, :]})

  def gen_update_A(self, correct_tags, current_tags):
    A_update = np.zeros([4, 4], dtype=np.float64)
    init_A_update = np.zeros([4], dtype=np.float64)
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

  def cal_sentence_loss(self, sentence, tags, length):
    sentence_embeds = self.sess.run(self.lookup_op, feed_dict={self.sentence_holder: sentence}).reshape(
      [length, self.concat_embed_size]).T
    sentence_scores = self.sess.run(self.word_score, feed_dict={self.x: sentence_embeds})

    init_A_val = self.init_A.eval(session=self.sess)
    A_val = self.A.eval(session=self.sess)
    current_tags = self.viterbi(sentence_scores, A_val, init_A_val)  # 当前参数下的最优路径
    loss = 0.0
    before_corr = 0
    before_cur = 0
    for index, (cur_tag, corr_tag, scores) in enumerate(zip(current_tags, tags, sentence_scores.T)):
      if index == 0:
        loss += scores[corr_tag] + init_A_val[corr_tag] - scores[cur_tag] - init_A_val[cur_tag]
      else:
        loss += scores[corr_tag] + A_val[before_corr, corr_tag] - scores[cur_tag] - A_val[before_cur, cur_tag]
      before_cur = cur_tag
      before_corr = corr_tag

    return math.fabs(loss)

  def sentence2index(self, sentence):
    index = []
    for word in sentence:
      if word not in self.dictionary:
        index.append(0)
      else:
        index.append(self.dictionary[word])

    return index

  def index2seq(self, indices):
    ext_indices = [1] * self.skip_window
    ext_indices.extend(indices + [2] * self.skip_window)
    seq = []
    for index in range(self.skip_window, len(ext_indices) - self.skip_window):
      seq.append(ext_indices[index - self.skip_window: index + self.skip_window + 1])

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

  def seg(self, sentence, model_path='model/model.ckpt', debug=False):
    dtype = tf.float64
    tf.reset_default_graph()
    x = tf.placeholder(dtype, shape=[self.concat_embed_size, None], name='x')
    # embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0, dtype=tf.float64),
    #                         tf.float64, name='embeddings')
    embeddings = tf.Variable(np.load('data/dnn/embeddings.npy'), dtype, name='embeddings')
    # w2 = tf.Variable(
    #  tf.truncated_normal([self.h, self.concat_embed_size], stddev=1.0 / math.sqrt(self.concat_embed_size),
    #                      dtype=tf.float64), dtype=tf.float64, name='w2')
    w2 = tf.Variable(np.load('data/dnn/w2.npy'), dtype, name='w2')
    # b2 = tf.Variable(tf.zeros([self.h, 1], dtype=tf.float64), dtype=tf.float64, name='b2')
    b2 = tf.Variable(np.load('data/dnn/b2.npy'), dtype, name='b2')
    # w3 = tf.Variable(
    #  tf.truncated_normal([self.tags_count, self.h], stddev=1.0 / math.sqrt(self.concat_embed_size), dtype=tf.float64),
    #  dtype=tf.float64, name='w3')
    w3 = tf.Variable(np.load('data/dnn/w3.npy'), dtype, name='w3')
    # b3 = tf.Variable(tf.zeros([self.tags_count, 1], dtype=tf.float64), dtype=tf.float64, name='b3')
    b3 = tf.Variable(np.load('data/dnn/b3.npy'), dtype, name='b3')
    word_score = tf.matmul(w3, tf.sigmoid(tf.matmul(w2, x) + b2)) + b3
    # A = tf.Variable(
    #  [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=tf.float64, name='A')
    A = tf.Variable(np.load('data/dnn/A.npy'), dtype, name='A')
    # init_A = tf.Variable([1, 1, 0, 0], dtype=tf.float64, name='init_A')
    init_A = tf.Variable(np.load('data/dnn/init_A.npy'), dtype, name='init_A')
    params = [embeddings, A, init_A, w2, w3, b2, b3]
    saver = tf.train.Saver(var_list=params)
    with tf.Session() as sess:
      saver.restore(sess, model_path)
      # tf.global_variables_initializer().run()
      # saver.save(sess,'model/model.ckpt')
      seq = self.index2seq(self.sentence2index(sentence))
      sentence_embeds = tf.nn.embedding_lookup(embeddings, seq).eval().reshape(
        [len(sentence), self.concat_embed_size]).T
      sentence_scores = sess.run(word_score, feed_dict={x: sentence_embeds})
      init_A_val = init_A.eval()
      A_val = A.eval()

      current_tags = self.viterbi(sentence_scores, A_val, init_A_val)
      if debug:
        w3v = w3.eval().T.tolist()
        # print(w3v)
        file = open('tmp/w3.txt', 'w')

        for i, v in enumerate(w3v):
          v = list(map(lambda f: str(f), v))
          file.write(' '.join(v) + '\n')
        file.close()

        w2v = w2.eval().T.tolist()
        # print(w3v)
        file = open('tmp/w22.txt', 'w')

        for i, v in enumerate(w2v):
          v = list(map(lambda f: str(f), v))
          file.write(' '.join(v) + '\n')
        file.close()

      return self.tags2words(sentence, current_tags), current_tags


if __name__ == '__main__':
  embed_size = 50
  cws = SegDNN(constant.VOCAB_SIZE, embed_size, constant.DNN_SKIP_WINDOW)
  cws.train()
  # print(cws.seg('小明来自南京师范大学'))
  # print(cws.seg('小明是上海理工大学的学生'))
  # print(cws.seg('迈向充满希望的新世纪'))
