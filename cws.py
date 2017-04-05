import tensorflow as tf
import math
import numpy as np
import sys
from datetime import datetime
from prepare_data import read_train_data, build_dataset_from_raw


class CWS:
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
      self.x = tf.placeholder(tf.float32, shape=[self.concat_embed_size, 1], name='x')
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
      # init_A = [[0.5,0.5,0,0],[1,0,0,0.15],[0,0,0.1,0],[1,0.01,0,0]]
      self.A = tf.Variable(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=tf.float32)

      param_list = [self.w2, self.w3, self.b2, self.b3]
      paramp_list = [self.w2p, self.w3p, self.b2p, self.b3p]
      update_list = [self.w2.assign_add(self.w2p), self.w3.assign_add(self.w3p), self.b2.assign_add(self.b2p),
                     self.b3.assign_add(self.b3p)]
      param_grad = [[0, 0, 0, 0, 0]] * 4
      for w_index, w in enumerate(self.word_scores):
        for p_index, p in enumerate(param_list):
          param_grad[w_index][p_index] = tf.gradients(w, p)
        param_grad[w_index][len(param_list)] = tf.gradients(w, self.x)

      saver = tf.train.Saver([self.embeddings, self.A].extend(param_list))

      with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        init.run()
        times = 0.0
        # 对每局句子进行参数更新
        for sentence_index, sentence in enumerate(words_batch):
          start = datetime.now().timestamp()
          print('s:' + str(sentence_index))

          sentence_embeds = tf.reshape(tf.nn.embedding_lookup(self.embeddings, sentence),
                                       [len(sentence), self.concat_embed_size, 1]).eval()
          sentence_scores = np.array(sess.run(self.word_score, feed_dict={self.x: sentence_embeds[0]}).T,
                                     dtype=np.float32)

          for embed in sentence_embeds[1:, :]:
            sentence_scores = np.append(sentence_scores, sess.run(self.word_score, feed_dict={self.x: embed}).T, 0)

          init_A_val = np.array(self.A.eval()[0])
          A_val = np.array(self.A.eval()[1:])
          current_tags = self.viterbi(sentence_scores, A_val, init_A_val)
          diff_tags = np.subtract(tags_batch[sentence_index], current_tags)

          for diff_index, diff_val in enumerate(diff_tags):
            if diff_val != 0:
              pos_grad_index = tags_batch[sentence_index][diff_index]
              neg_grad_index = current_tags[diff_index]
              for param_index, param in enumerate(param_list):
                self.update_param(param, param_grad[pos_grad_index][param_index], self.x, sentence_embeds[diff_index],
                                  1, update_list[param_index],paramp_list[param_index],sess)
                self.update_param(param, param_grad[neg_grad_index][param_index], self.x, sentence_embeds[diff_index],
                                  -1,update_list[param_index],paramp_list[param_index], sess)

              # start = datetime.now().timestamp()
              grad_x_pos_val = sess.run(param_grad[pos_grad_index][len(param_list)],
                                        feed_dict={self.x: sentence_embeds[diff_index]})
              grad_x_neg_val = sess.run(param_grad[neg_grad_index][len(param_list)],
                                        feed_dict={self.x: sentence_embeds[diff_index]})
              self.update_embeddings(self.embeddings, sentence[diff_index], 1, grad_x_pos_val[0])
              self.update_embeddings(self.embeddings, sentence[diff_index], -1, grad_x_neg_val[0])
              ss1 = datetime.now().timestamp()
              if diff_index == 0:
                tf.scatter_nd_add(self.A, [[0, tags_batch[sentence_index][diff_index]]], [self.alpha])
                tf.scatter_nd_add(self.A, [[0, current_tags[diff_index]]], [-self.alpha])
              else:
                before = tags_batch[sentence_index][diff_index - 1]
                tf.scatter_nd_add(self.A, [[before, tags_batch[sentence_index][diff_index]]], [self.alpha])
                tf.scatter_nd_add(self.A, [[current_tags[diff_index - 1], current_tags[diff_index]]], [-self.alpha])
              del grad_x_neg_val
              del grad_x_pos_val
              # del current_tags
              # del diff_tags
              # print(datetime.now().timestamp() - ss1)
          del sentence_embeds
          del sentence_scores
          print(datetime.now().timestamp() - start)
          times += datetime.now().timestamp() - start
          if sentence_index > 100:
            break
        print(times / 60)
        saver.save(sess, 'tmp/model.ckpt')

  def update_embeddings(self, embeddings, indices, delta_grad, val):
    # start = datetime.now().timestamp()
    tf.scatter_nd_add(embeddings, np.expand_dims(indices, 1),
                      (self.alpha * delta_grad * val).reshape(3, self.embed_size))
    # print(datetime.now().timestamp()-start)

  def update_param(self, param, grad, x, x_val, delta_grad, op,p,sess):
    # start = datetime.now().timestamp()
    grad_val = sess.run(grad, feed_dict={x: x_val})
    # print('g')
    # print(datetime.now().timestamp() - start)
    #start = datetime.now().timestamp()
    sess.run(op, {p:self.alpha * delta_grad * grad_val[0]})
    #param.assign_add(self.alpha * delta_grad * grad_val[0])
    # tf.assign_add(param, self.alpha * delta_grad * grad_val[0])
    # print('a')
    #print(datetime.now().timestamp() - start)

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


if __name__ == '__main__':
  vocab_size = 3500
  embed_size = 50
  skip_window = 1
  # sentences = open('sentences.txt').read().splitlines()
  # build_dataset(sentences,vocab_size)
  cws = CWS(vocab_size, embed_size, skip_window)
  cws.train()
  # print(cws.seg('我是谁'))
  # cws.write_data(vocab_size, skip_window, 'word.txt', 'label.txt')
