import tensorflow as tf
import numpy as np
import collections

def generate_batch(sentences):
  pass


def train(vocab_size,embed_size,skip_window,data):
  tags = [0,1,2,3]
  graph = tf.Graph()

  with graph.as_default():
    train_inputs = tf.placeholder(tf.int32)
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings,[[0,1],[1,2],[2,3]])
    init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
      r = sess.run(init)
      a = np.array(sess.run(tf.reshape(embed,[3,200])))
      print(a.shape)


if __name__ == '__main__':
  vocab_size = 200
  embed_size = 100
  skip_window = 1
  sentences = open('sentences.txt').read().splitlines()
  # build_dataset(sentences,vocab_size)
  train(vocab_size,embed_size,1,'')



