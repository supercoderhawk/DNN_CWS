import tensorflow as tf
import math
import numpy as np
import sys
# from datetime import datetime
import time
from prepare_data import read_train_data, build_dataset_from_raw


class SegDNN:
  def __init__(self, vocab_size, embed_size, skip_window):
    self.TAG_MAPS = np.array([[0, 1], [2, 3], [2, 3], [0, 1]], dtype=np.int32)
    self.dictionary = None
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.skip_window = skip_window
    self.alpha = 0.02
    self.tags = [0, 1, 2, 3]
    self.tags_count = len(self.tags)
    self.window_length = 2 * self.skip_window + 1
    self.concat_embed_size = self.embed_size * self.window_length
    # self.generate_batch()
    self.x = None
    self.g = 0
    self.a = 0
    self.grad_params = None
    self.params = None
    self.grad_params = [0] * 4
    self.grad_word_score = [0] * 4
    self.update_ops = None
    self.map_matrix = None
    self.grad_embed = None
    self.update_embed_op = None
    self.param_holders = None

    self.map_matrices = self.gen_map_matrix()

  def generate_batch(self):
    """
    产生用于训练的数据
    :param vocab_size:
    :param skip_window:
    :return:
    """
    sentences, vocab_index, label_index, count, self.dictionary = read_train_data(
      vocab_size)
    words_batch = []
    label_batch = []
    window_length = 2 * skip_window + 1
    for i, sentence in enumerate(vocab_index):
      if len(sentence) > window_length:
        sentence_batch = []
        for j, _ in enumerate(vocab_index[i]):
          if j == 0:
            sentence_batch.append([0] + sentence[j:j + skip_window + 1])
          elif j == len(vocab_index[i]) - 1:
            sentence_batch.append(sentence[j - skip_window:j + 1] + [0])
          else:
            sentence_batch.append(sentence[j - skip_window:j + skip_window + 1])
        words_batch.append(sentence_batch)
        label_batch.append(label_index[i])

    return words_batch, label_batch

  def read_data(self, word_file_name, label_file_name, skip_window):
    word_file = open(word_file_name, 'r', encoding='utf-8')
    label_file = open(label_file_name, 'r', encoding='utf-8')
    words = word_file.read().splitlines()
    labels = label_file.read().splitlines()
    words_batch = []
    label_batch = []
    window_length = 2 * skip_window + 1

    for word in words:
      word_list = list(map(int, word.split(' ')))
      words_batch.append(np.array(word_list).reshape([len(word_list) // window_length, window_length]))
    for label in labels:
      label_batch.append(list(map(int, label.split(' '))))
    word_file.close()
    label_file.close()
    return np.array(words_batch), np.array(label_batch)

  def write_data(self, vocab_size, skip_window, word_file_name, label_file_name):
    words_batch, label_batch = self.generate_batch()
    word_file = open(word_file_name, 'w', encoding='utf-8')
    label_file = open(label_file_name, 'w', encoding='utf-8')
    for index, word in enumerate(words_batch):
      word = np.array(word).reshape([3 * len(word)]).tolist()
      word_file.write(' '.join(map(str, word)) + '\n')
    for label in label_batch:
      label_file.write(' '.join(map(str, label)) + '\n')
    word_file.close()
    label_file.close()

  def gen_map_matrix(self):
    map_matrix = np.zeros([4, 4, 4, 4], dtype=np.float32)
    for i in range(4):
      for j in range(4):
        m = np.zeros([4, 4], dtype=np.float32)
        m[i, i] = 1
        m[j, j] = -1
        map_matrix[i, j, ...] = m

    return map_matrix

  def train(self, word_file_name='word.txt', label_file_name='label.txt'):
    """
    用于训练模型
    :param vocab_size:
    :param embed_size:
    :param skip_window:
    :return:
    """

    graph = tf.Graph()
    words_batch, tags_batch = self.read_data(word_file_name, label_file_name, self.skip_window)
    print('start...')
    # alpha = 0.02
    h = 300

    with graph.as_default():
      merged = tf.summary.merge_all()

      self.x = tf.placeholder(tf.float32, shape=[self.concat_embed_size, 1], name='x')
      self.map_matrix = tf.placeholder(tf.float32, shape=[4, 4], name='mm')
      self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='embeddings')
      self.w2 = tf.Variable(
        tf.truncated_normal([h, self.concat_embed_size], stddev=1.0 / math.sqrt(self.concat_embed_size)),
        name='w2')
      self.w2p = tf.placeholder(tf.float32, self.w2.get_shape())
      self.b2 = tf.Variable(tf.zeros([h, 1]), name='b2')
      self.b2p = tf.placeholder(tf.float32, self.b2.get_shape())
      self.w3 = tf.Variable(tf.truncated_normal([self.tags_count, h], stddev=1.0 / math.sqrt(self.concat_embed_size)),
                            name='w3')
      self.w3p = tf.placeholder(tf.float32, self.w3.get_shape())
      self.b3 = tf.Variable(tf.zeros([self.tags_count, 1]), name='b3')
      self.b3p = tf.placeholder(tf.float32, self.b3.get_shape())
      self.word_score = tf.matmul(self.w3, tf.sigmoid(tf.matmul(self.w2, self.x) + self.b2)) + self.b3
      self.word_scores = tf.split(self.word_score, len(self.tags))

      self.params = [self.w2, self.w3, self.b2, self.b3]
      for i in range(4):
        self.grad_params[i] = tf.gradients(tf.matmul(self.map_matrix, self.word_score), self.params[i])
      self.grad_embed = tf.gradients(tf.matmul(self.map_matrix, self.word_score), self.x)

      # init_A = [[0.5,0.5,0,0],[1,0,0,0.15],[0,0,0.1,0],[1,0.01,0,0]]
      self.A = tf.Variable(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=tf.float32)
      self.embedp = tf.placeholder(tf.float32, shape=[self.window_length, self.embed_size])
      self.embed_index = tf.placeholder(tf.int32, shape=[self.window_length, 1])
      # print(self.word_scores)

      self.param_holders = [self.w2p, self.w3p, self.b2p, self.b3p]
      self.update_ops = [self.w2.assign_add(self.w2p), self.w3.assign_add(self.w3p), self.b2.assign_add(self.b2p),
                         self.b3.assign_add(self.b3p)]
      self.update_embed_op = tf.scatter_nd_add(self.embeddings, self.embed_index,
                                               tf.reshape(self.embedp, [self.window_length, self.embed_size]))

      param_grad = [[0] * 5] * 4
      param_gradp = [[0] * 5] * 4
      self.grad_word_scores = [[0] * 4] * 4
      for w_index, w in enumerate(self.word_scores):
        # self.grad_word_score[w_index] = tf.gradients(self.word_score, param_list[w_index])
        for p_index, p in enumerate(self.params):
          param_grad[w_index][p_index] = tf.gradients(w, p)

          # paramp_list[w_index][p_index] =
        param_grad[w_index][len(self.params)] = tf.gradients(w, self.x)

      # print(tf.gradients(self.word_score,self.w2))

      saver = tf.train.Saver([self.embeddings, self.A].extend(self.params))

      with tf.Session(graph=graph) as sess:
        # merged = tf.merge_all_summaries()
        train_writer = tf.summary.FileWriter('logs', sess.graph)
        init = tf.global_variables_initializer()
        init.run()
        times = 0.0
        # sess.graph.finalize()
        # 对每局句子进行参数更新
        for sentence_index, sentence in enumerate(words_batch):
          start = time.time()
          print('s:' + str(sentence_index))
          # print(tf.nn.embedding_lookup(self.embeddings, sentence))
          sentence_embeds = tf.nn.embedding_lookup(self.embeddings, sentence).eval().reshape(
            [len(sentence), self.concat_embed_size, 1])
          # sentence_embeds = tf.reshape(tf.nn.embedding_lookup(self.embeddings, sentence),
          #                             [len(sentence), self.concat_embed_size, 1]).eval()
          sentence_scores = np.array(sess.run(self.word_score, feed_dict={self.x: sentence_embeds[0]}).T,
                                     dtype=np.float32)

          for embed in sentence_embeds[1:, :]:
            sentence_scores = np.append(sentence_scores, sess.run(self.word_score, feed_dict={self.x: embed}).T, 0)

          A_tolVal = self.A.eval()
          init_A_val = np.array(A_tolVal[0])
          A_val = np.array(A_tolVal[1:])
          current_tags = self.viterbi(sentence_scores, A_val, init_A_val)
          diff_tags = np.subtract(tags_batch[sentence_index], current_tags)
          update_index = np.where(diff_tags != 0)[0]

          update_pos_index = np.array(tags_batch[sentence_index], dtype=np.int32)[update_index]
          update_neg_index = current_tags[update_index]
          update_embed = sentence_embeds[update_index]
          sentence_matrix = self.gen_sentence_map_matrix(update_pos_index, update_neg_index)
          self.update_params(sentence_matrix, update_embed, sentence[update_index], sess)
          for diff_index, diff_val in enumerate(diff_tags):
            if diff_val != 0:

              if diff_index == 0:
                tf.scatter_nd_add(self.A, [[0, tags_batch[sentence_index][diff_index]]], [self.alpha])
                tf.scatter_nd_add(self.A, [[0, current_tags[diff_index]]], [-self.alpha])
              else:
                before = tags_batch[sentence_index][diff_index - 1]
                AA = tf.scatter_nd_add(self.A, [[before, tags_batch[sentence_index][diff_index]]], [self.alpha])
                BB = tf.scatter_nd_add(self.A, [[current_tags[diff_index - 1], current_tags[diff_index]]],
                                       [-self.alpha])


          print(time.time() - start)
          times += time.time() - start
          # train_writer.add_graph(graph)
          # train_writer.add_run_metadata(sess.)
          if sentence_index > 10:
            break
        print(times / 60)
        print(self.g)
        print(self.a)
        train_writer.flush()
        saver.save(sess, 'tmp/model.ckpt')

  def update_params(self, sen_matrix, embeds, embed_index, sess):
    length = len(sen_matrix)
    for i in range(length):
      # 更新w和b共4个参数
      for p_index, param in enumerate(self.grad_params):
        grad = sess.run(param, {self.map_matrix: sen_matrix[i], self.x: embeds[i]})[0]
        sess.run(self.update_ops[p_index], {self.param_holders[p_index]: self.alpha * grad})
      # 更新查找表
      grad = sess.run(self.grad_embed, {self.map_matrix: sen_matrix[i], self.x: embeds[i]})[0]
      sess.run(self.update_embed_op, {self.embedp: np.reshape(grad, [self.window_length, self.embed_size]),
                                      self.embed_index: np.expand_dims(embed_index[i], 1)})

  def update_embeddings(self, embeddings, indices, delta_grad, val):
    start = time.time()
    tf.scatter_nd_add(embeddings, np.expand_dims(indices, 1),
                      (self.alpha * delta_grad * val).reshape(3, self.embed_size))
    print(time.time() - start)

  def update_param_ex(self, param_index, pos, neg, x_val, op, p, sess):
    start = time.time()
    # grad_val = sess.run(self.grad_word_score[param_index],{self.x:x_val})[0]# sess.run(grad, {self.x: x_val})
    grad_val = sess.run(self.grad_word_score[param_index], {self.x: x_val})[0]
    # print(self.grad_word_score)
    # k = None
    # print(tf.gradients(self.grad_word_score, self.w2))
    # print(k)
    # print(self.word_scores)
    # print('g')
    # print(time.time() - start)
    # start = time.time()
    # g = sess.run(self.grad_word_score[0], {self.x: x_val})
    # print(self.word_score.shape)
    # print(self.w2.get_shape())
    # print(type(x_val))
    # g  = sess.run(tf.gradients(self.word_score, self.w2),{self.x: x_val})
    # print(time.time()-start)
    # print(g[0].shape)
    print(grad_val.shape)
    if (time.time() - start == 0):
      # print('gg')
      self.g += 1
    start = time.time()
    sess.run(op, {p: self.alpha * (grad_val[pos] - grad_val[neg])})
    # param.assign_add(self.alpha * delta_grad * grad_val[0])
    # tf.assign_add(param, self.alpha * delta_grad * grad_val[0])
    # print('a')
    # print(datetime.now().timestamp() - start)
    if (time.time() - start == 0):
      self.a += 1

  def viterbi(self, emission, A, init_A):
    """
    维特比算法的实现，
    :param emission: 发射概率矩阵，对应于本模型中的分数矩阵
    :param A: 转移概率矩阵
    :return:
    """

    path = np.array([[0], [1]], dtype=np.int32)
    print(emission.shape)
    path_score = np.array([[init_A[0] + emission[0, 0]], [init_A[1] + emission[0, 1]]], dtype=np.float32)

    for line_index in range(1, emission.shape[0]):
      last_index = path[:, -1]
      cur_index = self.TAG_MAPS[last_index]  # 当前所有路径的可选的标记矩阵，2x2
      cur_res = A[last_index, cur_index] + emission[line_index, cur_index] + np.expand_dims(path_score[:, -1], 1)
      cur_max_index = np.argmax(cur_res, 1)
      path = np.insert(path, [path.shape[1]], np.expand_dims(np.choose(cur_max_index, cur_index.T), 1), 1)
      path_score = np.insert(path_score, [path_score.shape[1]], np.expand_dims(np.choose(cur_max_index, cur_res.T), 1),
                             1)

    return path[np.argmax(path_score[:, -1]), :]

  def gen_sentence_map_matrix(self, pos_index, neg_index):
    length = len(pos_index)
    sen_matrix = np.zeros([length, 4, 4], dtype=np.int32)
    for index in range(length):
      sen_matrix[index, ...] = self.map_matrices[pos_index[index], neg_index[index]]

    return sen_matrix

  def sentence2index(self, sentence):
    index = []
    for word in sentence:
      if word not in self.dictionary:
        index.append(0)
      else:
        index.append(self.dictionary[word])

    return index

  def index2seq(self, indices):
    ext_indices = [0] * self.skip_window
    ext_indices.extend(indices + [0] * self.skip_window)
    seq = []
    for index in range(self.skip_window, len(ext_indices) - self.skip_window):
      seq.append(ext_indices[index - self.skip_window: index + self.skip_window + 1])

    return seq

  def tags2words(self, sentence, tags):
    print(tags)
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

  def seg(self, sentence):
    graph = tf.Graph()
    h = 300
    with graph.as_default():
      x = tf.placeholder(tf.float32, shape=[self.concat_embed_size, 1], name='x')
      embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='embeddings')
      w2 = tf.Variable(tf.truncated_normal([h, self.concat_embed_size], stddev=1.0 / math.sqrt(self.concat_embed_size)),
                       name='w2')
      b2 = tf.Variable(tf.zeros([h, 1]), name='b2')
      w3 = tf.Variable(tf.truncated_normal([self.tags_count, h], stddev=1.0 / math.sqrt(self.concat_embed_size)),
                       name='w3')
      b3 = tf.Variable(tf.zeros([self.tags_count, 1]), name='b3')
      word_score = tf.matmul(w3, tf.sigmoid(tf.matmul(w2, x) + b2)) + b3
      A = tf.Variable(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=tf.float32)
      # word_scores = tf.split(word_score, len(tags))
      saver = tf.train.Saver()

      with tf.Session(graph=graph) as sess:
        saver.restore(sess, 'tmp/model.ckpt')
        seq = self.index2seq(self.sentence2index(sentence))
        sentence_embeds = tf.reshape(tf.nn.embedding_lookup(embeddings, seq),
                                     [len(seq), self.concat_embed_size, 1]).eval()
        sentence_scores = np.array(sess.run(word_score, feed_dict={x: sentence_embeds[0]}).T, dtype=np.float32)

        for embed in sentence_embeds[1:, :]:
          sentence_scores = np.append(sentence_scores, sess.run(word_score, feed_dict={x: embed}).T, 0)
        init_A_val = np.array(A.eval()[0])
        A_val = np.array(A.eval()[1:])
        print(sentence_scores)
        current_tags = self.viterbi(sentence_scores, A_val, init_A_val)
        return self.tags2words(sentence, current_tags)


if __name__ == '__main__':
  vocab_size = 3500
  embed_size = 50
  skip_window = 1
  # sentences = open('sentences.txt').read().splitlines()
  # build_dataset(sentences,vocab_size)
  cws = SegDNN(vocab_size, embed_size, skip_window)
  cws.train()
  # print(cws.seg('我们是朋友'))
  # cws.write_data(vocab_size, skip_window, 'word.txt', 'label.txt')
