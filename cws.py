import tensorflow as tf
import math
import numpy as np
import sys
from prepare_data import read_train_data


def generate_batch(vocab_size, skip_window):
  """
  产生用于训练的数据
  """
  sentences, vocab_index, label_index, count, dictionary = read_train_data(
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


def read_data(word_file_name, label_file_name, skip_window):
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
  return words_batch, label_batch


def write_data(vocab_size, skip_window, word_file_name, label_file_name):
  words_batch, label_batch = generate_batch(vocab_size, skip_window)
  word_file = open(word_file_name, 'w', encoding='utf-8')
  label_file = open(label_file_name, 'w', encoding='utf-8')
  for index, word in enumerate(words_batch):
    word = np.array(word).reshape([3 * len(word)]).tolist()
    word_file.write(' '.join(map(str, word)) + '\n')
  for label in label_batch:
    label_file.write(' '.join(map(str, label)) + '\n')
  word_file.close()
  label_file.close()


def train(vocab_size, embed_size, skip_window, word_file_name='word.txt',
          label_file_name='label.txt'):
  """
  用于训练模型
  :param vocab_size:
  :param embed_size:
  :param skip_window:
  :param data:
  :return:
  """
  tags = [0, 1, 2, 3]
  tags_count = len(tags)
  window_length = 2 * skip_window + 1
  graph = tf.Graph()
  words_batch, tags_batch = read_data(word_file_name, label_file_name, skip_window)
  print('start...')
  alpha = 0.02
  h = 300
  with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[window_length * embed_size, 1])
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
    w2 = tf.Variable(
      tf.truncated_normal([h, embed_size * window_length], stddev=1.0 / math.sqrt(embed_size * window_length)))
    b2 = tf.Variable(tf.zeros([h, 1]))

    w3 = tf.Variable(tf.truncated_normal([tags_count, h], stddev=1.0 / math.sqrt(embed_size * window_length)))
    b3 = tf.Variable(tf.zeros([tags_count, 1]))

    word_score = tf.matmul(w3, tf.sigmoid(tf.matmul(w2, x) + b2)) + b3
    word_scores = tf.split(word_score, len(tags))
    # init_A = [[0.5,0.5,0,0],[1,0,0,0.15],[0,0,0.1,0],[1,0.01,0,0]]
    A = tf.Variable(
      [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])

    param_list = [w2, w3, b2, b3]
    # grad_list = [grad_w2, grad_w3, grad_b2, grad_b3]

    with tf.Session(graph=graph) as sess:
      init = tf.global_variables_initializer()
      init.run()
      # 对每局句子进行参数更新
      for sentence_index, sentence in enumerate(words_batch):
        print('s:' + str(sentence_index))
        sentence_embeds = tf.reshape(tf.nn.embedding_lookup(embeddings, sentence),
                                     [len(sentence), window_length * embed_size, 1]).eval()

        sentence_scores = list()
        for embed in sentence_embeds:
          sentence_scores.append(tf.transpose(sess.run(word_score, feed_dict={x: embed})).eval())

        current_tags = viterbi(np.reshape(sentence_scores, [len(sentence), len(tags), 1]), A.eval())

        diff_tags = np.subtract(np.array(tags_batch[sentence_index]), np.array(current_tags)).tolist()

        for diff_index, diff_val in enumerate(diff_tags):
          if diff_val != 0:
            pos_grad = word_scores[tags_batch[sentence_index][diff_index]]
            neg_grad = word_scores[current_tags[diff_index]]
            for param in param_list:
              update_param(param, tf.gradients(pos_grad, param), x, sentence_embeds[diff_index], alpha, 1, sess)
              update_param(param, tf.gradients(neg_grad, param), x, sentence_embeds[diff_index], alpha, -1, sess)

            grad_x_pos_val = sess.run(tf.gradients(pos_grad, x),feed_dict={x: sentence_embeds[diff_index]})
            grad_x_neg_val = sess.run(tf.gradients(neg_grad, x),feed_dict={x: sentence_embeds[diff_index]})
            update_embeddings(embeddings, sentence[diff_index], alpha, 1,grad_x_pos_val[0], embed_size)
            update_embeddings(embeddings, sentence[diff_index], alpha, -1,grad_x_neg_val[0], embed_size)

            if diff_index == 0:
              tf.scatter_nd_add(A, [[0, tags_batch[sentence_index][diff_index]]], [alpha])
              tf.scatter_nd_add(A, [[0, current_tags[diff_index]]], [-alpha])
            else:
              before = tags_batch[sentence_index][diff_index - 1]
              tf.scatter_nd_add(A, [[before, tags_batch[sentence_index][diff_index]]], [alpha])
              tf.scatter_nd_add(A, [[current_tags[diff_index - 1], current_tags[diff_index]]], [-alpha])


def update_embeddings(embeddings, indices, alpha, delta_grad, val, embed_size):
  embeddings_val = embeddings.eval()
  offset = -1 * embed_size
  for index in indices:
    offset += embed_size
    for i, v in enumerate(embeddings_val[index]):
      embeddings_val[index][i] += alpha * delta_grad * val[offset + i]
  tf.assign(embeddings, embeddings_val)


def update_param(param, grad, x, x_val, alpha, delta_grad, sess):
  grad_val = sess.run(grad, feed_dict={x: x_val})

  tf.assign(param,np.add(param.eval(), alpha * delta_grad * np.array(grad_val)[0]))


def viterbi(emission, A):
  """
  维特比算法的实现，
  :param emission: 发射概率矩阵，对应于本模型中的分数矩阵
  :param A: 转移概率矩阵
  :return:
  """

  tags = [1, 2, 3, 4]
  path = {0: [1], 1: [2]}
  path_score = {0: [A[0][0] + emission[0][0]], 1: [A[0][1] + emission[0][1]]}
  tag_maps = {1: [1, 2], 2: [3, 4], 3: [3, 4], 4: [1, 2]}

  for line_index, line in enumerate(emission):
    if (line_index == 0):
      continue

    for path_index, _ in enumerate(path):
      last_index = path[path_index][line_index - 1]
      tag_map = tag_maps[last_index]
      (score, tag) = max(
        [(path_score[path_index][line_index - 1] + emission[line_index][i - 1] + A[last_index][i - 1], i)
         for i in tag_map])
      path[path_index].append(tag)
      path_score[path_index].append(score)

  if path_score[0][len(emission) - 1] < path[1][len(emission) - 1]:
    return list(map(lambda x: x - 1, path[1]))
  else:
    return list(map(lambda x: x - 1, path[0]))


if __name__ == '__main__':
  vocab_size = 1000
  embed_size = 100
  skip_window = 1
  # sentences = open('sentences.txt').read().splitlines()
  # build_dataset(sentences,vocab_size)
  train(vocab_size, embed_size, skip_window)
  # write_data(vocab_size, skip_window, 'word.txt', 'label.txt')
