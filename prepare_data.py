# -*- coding: UTF-8 -*-
import collections
import re
from utils import strQ2B
import constant


class PrepareData:
  def __init__(self, input_file, output_words_file, output_labels_file, vocab_size):
    """
    构造函数
    :param input_file:  输入语料的完整文件路径 
    :param output_words_file: 输出的字符索引文件完整路径，字符索引文件中的内容是输入预料中每个字在词汇表中对应的索引
    :param output_label_file: 输出标签索引文件的完整路径，标签索引文件中的内容是输入语料中每个字对应的分词标签编号，采用SBIE标签，对应编号为0,1,2,3
    :param vocab_size: 词汇表的大小
    """
    self.input_file = input_file
    self.output_words_file = output_words_file
    self.output_labels_file = output_labels_file
    self.vocab_size = vocab_size  # 词汇表大小
    self.SPLIT_CHAR = '  '  # 分隔符：双空格
    self.sentences = self.read_sentences()  # 从输入文件中读取的句子列表
    self.words_index = []  # 语料文件中每个字对应的索引，以句子为单位
    self.labels_index = []  # 语料库中每个字对应的索引，采用SBIE标记，以句子为单位
    self.dictionary = {}  # 字符编号，从0开始，{'UNK':0,'STRT':'1','END':2,'我':3,'们':4}
    self.count = [['UNK', 0], ['STRT', 0],
                  ['END', 0]]  # 字符数量，其中'UNK'表示词汇表外的字符，'STAT'表示句子首字符之前的字符，'END'表示句子尾字符后面的字符，这两字符用于生成字的上下文

  def read_sentences(self):
    file = open(self.input_file, 'r', encoding='utf-8')
    content = file.read()
    sentences = re.sub('[ ]+', self.SPLIT_CHAR, strQ2B(content)).splitlines()  # 将词分隔符统一为双空格
    file.close()
    return sentences

  def build_basic_dataset(self):
    words = ''.join(self.sentences).replace(' ', '')
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - 3))

    for word, _ in self.count:
      self.dictionary[word] = len(self.dictionary)

    unk_count = 0
    # 给语料中的每个字标对应的序号
    for sentence in self.sentences:
      sentence = sentence.replace(' ', '')
      senData = []
      for word in sentence:
        if word in self.dictionary:
          index = self.dictionary[word]
        else:
          index = 0
          unk_count += 1
        senData.append(index)
      self.words_index.append(senData)
    self.count[0][1] = unk_count

  def build_corpus_dataset(self):
    empty = 0
    for sentence in self.sentences:
      sentence_label = []
      words = sentence.strip().split(self.SPLIT_CHAR)
      for word in words:
        l = len(word)
        if l == 0:
          empty += 1
          continue
        elif l == 1:
          sentence_label.append(0)
        else:
          sentence_label.append(1)
          sentence_label.extend([2] * (l - 2))
          sentence_label.append(3)
      self.labels_index.append(sentence_label)

  def build_exec(self):
    self.build_basic_dataset()
    self.build_corpus_dataset()
    words_file = open(self.output_words_file, 'w+', encoding='utf-8')
    labels_file = open(self.output_labels_file, 'w+', encoding='utf-8')

    for _, (words, labels) in enumerate(zip(self.words_index, self.labels_index)):
      words_file.write(' '.join(str(word) for word in words) + '\n')
      labels_file.write(' '.join(str(label) for label in labels) + '\n')
    words_file.close()
    labels_file.close()


if __name__ == '__main__':

  prepare_pku = PrepareData('corpus/pku_training.utf8', 'corpus/pku_training_words.txt', 'corpus/pku_training_labels.txt',
                        constant.VOCAB_SIZE)
  prepare_pku.build_exec()
  prepare_msr = PrepareData('corpus/msr_training.utf8', 'corpus/msr_training_words.txt', 'corpus/msr_training_labels.txt',
                        constant.VOCAB_SIZE)
  prepare_msr.build_exec()
