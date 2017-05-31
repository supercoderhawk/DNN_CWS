#-*- coding: UTF-8 -*-
from transform_data import TransformData
import constant
import numpy as np
import os

class TransformDataLSTM(TransformData):
  def __init__(self,gen=False):
    TransformData.__init__(self, 'corpus/dict.utf8', ['pku'])
    self.skip_window_left = constant.LSTM_SKIP_WINDOW_LEFT
    self.skip_window_right = constant.LSTM_SKIP_WINDOW_RIGHT
    #self.skip_window = self.skip_window_left + self.skip_window_right + 1
    self.words_batch_base_path = 'corpus/lstm/words_batch_'+str(self.skip_window_left)+'_'+str(self.skip_window_right)
    self.words_batch_path = self.words_batch_base_path + '.npy'
    self.labels_batch_base_path = 'corpus/lstm/labels_batch'
    self.labels_batch_path = self.labels_batch_base_path + '.npy'
    if not gen:
      if os.path.exists(self.words_batch_base_path+'.npy') and os.path.exists(self.labels_batch_base_path+'.npy'):
        self.words_batch = np.load(self.words_batch_path)
        self.labels_batch = np.load(self.labels_batch_path)
        return

    self.words_batch, self.labels_batch = self.generate_sentences_batch()

  def generate_sentences_batch(self):
    words_batch = []
    labels_batch = []

    for i, words in enumerate(self.words_index):
      if len(words) < max(self.skip_window_left,self.skip_window_right):
        continue

      extend_words = [1] * self.skip_window_left
      extend_words.extend(words)
      extend_words.extend([2] * self.skip_window_right)
      word_batch = list(map(lambda item: extend_words[item[0] - self.skip_window_left:item[0] + self.skip_window_right + 1],
                            enumerate(extend_words[self.skip_window_left:-self.skip_window_right], self.skip_window_left)))
      words_batch.append(np.array(word_batch,dtype=np.int32))
      labels_batch.append(np.array(self.labels_index[i],dtype=np.int32))

    return np.array(words_batch), np.array(labels_batch)

  def generate_exe(self):
    np.save(self.words_batch_base_path,self.words_batch)
    np.save(self.labels_batch_base_path,self.labels_batch)

  def generate_batch(self):
    pass

if __name__ == '__main__':
  trans = TransformDataLSTM(True)
  trans.generate_exe()