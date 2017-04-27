# -*- coding: UTF-8 -*-


class TransformData:
  def __init__(self, dict_path, corpuses):
    self.dict_path = dict_path
    self.dictionary = self.read_dictionary()
    self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    self.words_index = []
    self.labels_index = []
    if corpuses is not None or len(corpuses) != 0:
      for _, corpus in enumerate(corpuses):
        base_path = 'corpus/' + corpus + '_training'
        self.read_words(base_path + '_words.txt')
        self.read_labels(base_path + '_labels.txt')

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    return dictionary

  def read_words(self, path):
    file = open(path, 'r', encoding='utf-8')
    words = file.read().splitlines()
    for index, word in enumerate(words):
      self.words_index.append(list(map(int, word.split(' '))))
    file.close()

  def read_labels(self, path):
    file = open(path, 'r', encoding='utf-8')
    labels = file.read().splitlines()
    for label in labels:
      self.labels_index.append(list(map(int, label.split(' '))))
    file.close()

  def generate_batch(self):
    raise NotImplementedError('must implement generate batch function')
