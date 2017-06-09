# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
import constant
from transform_data_w2v import TransformDataW2V


class Word2Vec:
  def __init__(self, output, batch_size=128, num_skips=2, skip_window=1, vocab_size=constant.VOCAB_SIZE, embed_size=50,
               num_sampled=64, steps=100000):
    self.output = output
    self.batch_size = batch_size
    self.num_skips = num_skips
    self.skip_window = skip_window
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.num_sampled = num_sampled
    self.steps = steps
    self.tran = TransformDataW2V(self.batch_size, self.num_skips, self.skip_window)
    self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))

  def train(self):
    train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

    embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

    nce_weights = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size],
                          stddev=1.0 / math.sqrt(self.embed_size)))
    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                     num_sampled=self.num_sampled, num_classes=self.vocab_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      aver_loss = 0
      for step in range(self.steps):
        batch_inputs, batch_labels = self.tran.generate_batch()
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        aver_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            aver_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": ", aver_loss)
          aver_loss = 0
      np.save(self.output, self.embeddings.eval())

  def test(self):
    valid_dataset = [3021]
    norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
    normalized_embeddings = self.embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.abs(tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True))
    print(similarity.eval())
    pair = zip(range(self.vocab_size), similarity.eval()[0])
    spair = sorted(pair, key=lambda x: x[1])
    print(spair[0:10])


if __name__ == '__main__':
  w2v = Word2Vec('corpus/lstm/embeddings', embed_size=100)
  w2v.train()
