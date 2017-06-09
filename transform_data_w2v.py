# -*- coding: UTF-8 -*-
import numpy as np
import collections
import random
from transform_data import TransformData


class TransformDataW2V(TransformData):
  def __init__(self, batch_size, num_skips, skip_window):
    TransformData.__init__(self, 'corpus/dict.utf8', ['pku'])
    self.batch_size = batch_size
    self.num_skips = num_skips
    self.skip_window = skip_window
    self.data_index = 0
    self.span = 2 * self.skip_window + 1
    self.words = self.generate_words('sogou')
    self.word_count = len(self.words)

  def generate_words(self, name):
    if name == 'pku':
      return [item for sublist in self.words_index for item in sublist]
    elif name == 'sogou':
      with open('corpus/sogou.txt', 'r', encoding='utf8') as file:
        return self.sentence2index(file.read())

  def sentence2index(self, sentence):
    index = []
    for ch in sentence:
      if ch in self.dictionary:
        index.append(self.dictionary[ch])
      else:
        index.append(0)
    return index

  def generate_batch(self):
    batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
    span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(self.words[self.data_index])
      self.data_index = (self.data_index + 1) % self.word_count
    for i in range(self.batch_size // self.num_skips):
      target = self.skip_window  # target label at the center of the buffer
      targets_to_avoid = [self.skip_window]
      for j in range(self.num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * self.num_skips + j] = buffer[self.skip_window]
        labels[i * self.num_skips + j, 0] = buffer[target]
      buffer.append(self.words[self.data_index])
      self.data_index = (self.data_index + 1) % self.word_count
    # Backtrack a little bit to avoid skipping words in the end of a batch
    self.data_index = (self.data_index + self.word_count - span) % self.word_count
    return batch, labels
