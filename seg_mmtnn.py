# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
from seg_base import SegBase


class SegMMTNN(SegBase):
  def __init__(self, vocab_size, embed_size, skip_window):
    SegBase.__init__(self)
    self.skip_window = skip_window
    self.vocab_size = vocab_size
    self.window_size = 2 * self.skip_window + 1
    self.vec_length = self.window_size + 1
    self.embed_size = embed_size
    self.concat_size = self.vec_length * self.embed_size
    self.hidden_unit = 50
    self.r = 10
    self.x = tf.placeholder(dtype=tf.float32, shape=[self.concat_size, 1])
    self.w1 = tf.Variable(
      tf.truncated_normal([self.hidden_unit, self.concat_size], stddev=1.0 / math.sqrt(self.concat_size)))
    self.b1 = tf.Variable(tf.zeros([self.hidden_unit, 1]), dtype=tf.float32)
    self.w2 = tf.Variable(tf.random_uniform([self.tags_count, self.hidden_unit], -1.0, 1.0))
    self.b2 = tf.Variable(tf.zeros([self.tags_count, 1]), dtype=tf.float32)
    self.embeddings = tf.Variable(tf.random_uniform([self.tags_count, self.hidden_unit], -1.0, 1.0))
    self.L = tf.Variable(tf.random_normal([self.embed_size, self.tags_count], 0, 1))
    self.P = tf.Variable(tf.random_uniform([self.hidden_unit, self.concat_size, self.r], -1.0, 1.0))
    self.Q = tf.Variable(tf.random_uniform([self.hidden_unit, self.r, self.concat_size], -1.0, 1.0))
    self.word_score = None
    self.sess = None

  def train(self):
    self.sess = tf.Session()
    h = tf.add(tf.matmul(tf.matmul(tf.matmul(tf.transpose(self.x), self.P), self.Q), self.x),
                tf.matmul(self.w1, self.x))

    self.word_score = tf.add(tf.matmul(self.w2, tf.sigmoid(tf.add(h,self.b1))), self.b2)
    self.sess.close()

  def train_exe(self):
    pass

  def train_sentence(self):
    pass

if __name__ == '__main__':
  pass
