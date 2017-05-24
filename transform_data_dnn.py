# -*- coding: UTF-8 -*-
import numpy as np
import os
from transform_data import TransformData
import constant


class TransformDataDNN(TransformData):
  def __init__(self, skip_window,gen=False):
    """
    构造函数
    :param skip_window: 
    :param gen: 指定是否强制从数据文件生成所需的数据，若为True，强制生成，否则若存在已生成的文件，直接加载，默认为False 
    """
    TransformData.__init__(self, 'corpus/dict.utf8', ['pku'])
    self.skip_window = skip_window
    self.window_length = 2 * self.skip_window + 1
    self.words_batch_base_path = 'corpus/dnn/words_batch'
    self.words_batch_path = self.words_batch_base_path + '.npy'
    self.words_batch_flat_path = self.words_batch_base_path + '_flat.npy'
    self.labels_batch_base_path = 'corpus/dnn/labels_batch'
    self.labels_batch_flat_path = self.labels_batch_base_path + '_flat.npy'
    self.labels_batch_path = self.labels_batch_base_path + '.npy'
    if not gen:
      if os.path.exists(self.words_batch_path) and os.path.exists(self.labels_batch_path):
        self.words_batch = np.load(self.words_batch_path)
        self.labels_batch = np.load(self.labels_batch_path)
        self.words_batch_flat = np.load(self.words_batch_flat_path)
        self.labels_batch_flat = np.load(self.labels_batch_flat_path)
    else:
      self.words_batch, self.labels_batch = self.generate_sentences_batch()
      self.words_batch_flat,self.labels_batch_flat = self.generate_batch()
    self.words_count = len(self.labels_batch_flat) # 语料库中字符总个数
    self.context_count = self.words_count*self.window_length  # 生成的上下文词总长度
    self.whole_words_batch = self.words_batch_flat.reshape([self.words_count,self.window_length])
    self.whole_labels_batch = self.labels_batch_flat.reshape([self.words_count])

  def generate_sentences_batch(self):
    words_batch = []
    labels_batch = []
    for i, words in enumerate(self.words_index):
      if len(words) < self.skip_window:
        continue
      extend_words = [1] * self.skip_window
      extend_words.extend(words)
      extend_words.extend([2] * self.skip_window)
      word_batch = list(map(lambda item: extend_words[item[0] - self.skip_window:item[0] + self.skip_window + 1],
                            enumerate(extend_words[self.skip_window:-self.skip_window], self.skip_window)))
      words_batch.append(np.array(word_batch,dtype=np.int32))
      labels_batch.append(np.array(self.labels_index[i],dtype=np.int32))

    return np.array(words_batch), np.array(labels_batch)

  def generate_batch(self):
    words_batch = []
    labels_batch = []
    for _,(words,labels) in enumerate(zip(self.words_index,self.labels_index)):
      if len(words) < self.skip_window:
        return
      extend_words = [1] * self.skip_window
      extend_words.extend(words)
      extend_words.extend([2] * self.skip_window)
      word_batch = list(map(lambda item: extend_words[item[0] - self.skip_window:item[0] + self.skip_window + 1],
                            enumerate(extend_words[self.skip_window:-self.skip_window], self.skip_window)))
      words_batch.extend(np.array(word_batch,dtype=np.int32).flatten().tolist())
      labels_batch.extend(labels)

    return np.array(words_batch),np.array(labels_batch)


  def generate_exe(self):
    np.save(self.words_batch_base_path, self.words_batch)
    np.save(self.words_batch_flat_path, self.words_batch_flat)
    np.save(self.labels_batch_base_path, self.labels_batch)
    np.save(self.labels_batch_flat_path, self.labels_batch_flat)


if __name__ == '__main__':
  trans_dnn = TransformDataDNN(constant.DNN_SKIP_WINDOW,True)
  trans_dnn.generate_exe()
