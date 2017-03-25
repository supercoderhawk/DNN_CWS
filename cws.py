import tensorflow as tf
import math
import numpy as np
import sys
from prepare_data import read_train_data


def generate_batch(vocab_size, window_length):
  """
  产生用于训练的数据
  """
  # print(len(read_train_data(100)))
  # sys.exit(0)
  sentences, vocab_index, label_index, count, dictionary = read_train_data(
    vocab_size)
  words_batch = list()
  label_batch = list()
  for i in range(len(sentences)):
    if len(vocab_index) < window_length:
      continue
    else:
      sentence_batch = list()
      for j in range(len(vocab_index[i]) - window_length + 1):
        if j == 0:
          sentence_batch.append([0] + vocab_index[i][j:j + window_length - 1])
        elif j == len(vocab_index[i]) - window_length:
          sentence_batch.append(vocab_index[i][j:j + window_length - 1] + [0])
        else:
          sentence_batch.append(vocab_index[i][j:j + window_length])
      words_batch.append(sentence_batch)
      label_batch.append(label_index[i])

  return words_batch, label_batch


def train(vocab_size, embed_size, skip_window):
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
  words_batch, tags_batch = generate_batch(vocab_size, window_length)
  words_count = len(words_batch)
  alpha = 0.02
  h = 300
  with graph.as_default():
    x = tf.placeholder(tf.float32, shape = [window_length * embed_size, 1])
    # train_labels = tf.placeholder(tf.int32, shape=1)
    embeddings = tf.Variable(
      tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
    # embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # embed = tf.reshape(embed,[words_count,window_length * embed_size])
    w2 = tf.Variable(
      tf.truncated_normal([h, embed_size * window_length],
                          stddev = 1.0 / math.sqrt(embed_size * window_length)))
    b2 = tf.Variable(tf.zeros([h, 1]))

    w3 = tf.Variable(tf.truncated_normal([tags_count, h],
                                         stddev = 1.0 / math.sqrt(
                                           embed_size * window_length)))
    b3 = tf.Variable(tf.zeros([tags_count, 1]))

    word_score = tf.matmul(w3,
                           tf.sigmoid(tf.matmul(w2, x) + b2)) + b3
    #init_A = [[0.5,0.5,0,0],[1,0,0,0.15],[0,0,0.1,0],[1,0.01,0,0]]
    A = tf.Variable([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1],[1,1,0,0]])
    print(A.get_shape())

    #init_A = tf.Variable(tf.zeros([tags_count]))

    grad_w2 = tf.gradients(word_score, w2)
    grad_w3 = tf.gradients(word_score, w3)
    grad_b2 = tf.gradients(word_score, b2)
    grad_b3 = tf.gradients(word_score, b3)
    grad_x = tf.gradients(word_score, x)
    grad_A = tf.gradients(word_score, A)
    #grad_init_A = tf.gradients(word_score, init_A)

    with tf.Session(graph = graph) as sess:
      init = tf.global_variables_initializer()
      init.run()
      # scores = list()
      # 对每局句子进行参数更新
      for sentence_index, sentence in enumerate(words_batch):
        sentence_embeds = tf.reshape(
          tf.nn.embedding_lookup(embeddings, sentence),
          [len(sentence), window_length * embed_size, 1]).eval()
        sentence_scores = list()
        for embed in sentence_embeds:
          sentence_scores.append(tf.transpose(
            sess.run(word_score, feed_dict = {x: embed})).eval())
        # scores.append(sentence_scores)
        # print(sentence_scores)
        # print(viterbi(sentence_scores, A.eval(), init_A.eval()))
        current_tags = viterbi(
          np.reshape(sentence_scores, [len(sentence), len(tags), 1]), A.eval())
        print(current_tags)
        print(list(map(lambda x:x+1,tags_batch[sentence_index])))
        diff_tags = tags_batch[sentence_index] - current_tags
        # diff_words = list()
        # diff_A = list()
        for diff_index, diff_val in enumerate(diff_tags):
          if diff_val != 0:
            grad_w2_val = sess.run(grad_w2,
                                   feed_dict = {x: sentence_embeds[diff_index]})
            # print(list(grad_w2_val))
            w2 = tf.assign(w2, w2.eval() + alpha * np.array(grad_w2_val)[0])

            grad_w3_val = sess.run(grad_w3,
                                   feed_dict = {x: sentence_embeds[diff_index]})
            w3 = tf.assign(w3, w3.eval() + alpha * np.array(grad_w3_val)[0])

            grad_b2_val = sess.run(grad_b2,
                                   feed_dict = {x: sentence_embeds[diff_index]})
            b2 = tf.assign(b2, b2.eval() + alpha * np.array(grad_b2_val)[0])

            grad_b3_val = sess.run(grad_b3,
                                   feed_dict = {x: sentence_embeds[diff_index]})
            b3 = tf.assign(b3, b3.eval() + alpha * np.array(grad_b3_val)[0])

            grad_x_val = sess.run(grad_x,
                                  feed_dict = {x: sentence_embeds[diff_index]})
            # print(np.array(sentence_embeds[diff_index]).shape)
            # print(np.array(grad_x_val).shape)
            # print(np.array(x.eval()).shape)
            #x = tf.assign(x, sentence_embeds[diff_index] + alpha * np.array(
            #  grad_x_val))
            print(embeddings.get_shape())
            print(np.array(grad_x_val).shape)
            embeddings = tf.assign(embeddings, update_embeddings(embeddings.eval(),diff_index,grad_x_val[0]))
            #if diff_index == 0:
            #  grad_init_A_val = sess.run(grad_init_A,
            #                             feed_dict = {
            #                               x: sentence_embeds[diff_index]})
            # init_A = tf.assign(init_A,
            #                     init_A.eval() + alpha *
            #                     np.array(grad_init_A_val)[0])
            #else:
            grad_A_val = sess.run(grad_A,
                                  feed_dict = {
                                    x: sentence_embeds[diff_index]})
            A = tf.assign(A, A.eval() + alpha * np.array(grad_A_val)[0])


def update_embeddings(embeddings, index, val):
  for i, v in enumerate(embeddings[index]):
    embeddings[index][i] += val[index]

  return embeddings


def update_param(w2, w3, init_val, alpha, diff_c1, diff_c2):
  return init_val + alpha * diff_c1 * diff_c2


def viterbi(emission, A):
  """
  维特比算法的实现，
  :param emission: 发射概率矩阵，对应于本模型中的分数矩阵
  :param A: 转移概率矩阵
  :return:
  """
  seq = [{}]
  tags = [1, 2, 3, 4]
  path = {0: [0], 1: [1], 2: [2], 3: [3]}

  for i, val in enumerate(emission[0]):
    seq[0][i] = val + A[0][i]

  for line_index, line in enumerate(emission):
    if (line_index == 0):
      continue

    new_path = {}
    for val_index, val in enumerate(line):
      (max_val, tag) = max(
        [(val + emission[line_index - 1][i-1] + A[i][val_index], i) for i in
         tags])
      emission[line_index][val_index] = max_val
      new_path[val_index] = path[tag-1] + [val_index+1]

    path = new_path

  max_index, _ = max([(i-1, emission[len(emission) - 1][i-1]) for i in tags])
  return path[max_index]


if __name__ == '__main__':
  vocab_size = 1000
  embed_size = 100
  skip_window = 1
  # sentences = open('sentences.txt').read().splitlines()
  # build_dataset(sentences,vocab_size)
  train(vocab_size, embed_size, skip_window)
