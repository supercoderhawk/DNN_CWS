# -*- coding: UTF-8 -*-
import collections
import re
from utils import strQ2B
import constant


class PrepareData:
  def __init__(self, vocab_size, input_file, output_words_file, output_labels_file, dict_file, raw_file,input_dict=False):
    """
    构造函数
    :param vocab_size: 词汇表的大小
    :param input_file:  输入语料的完整文件路径 
    :param output_words_file: 输出的字符索引文件完整路径，字符索引文件中的内容是输入预料中每个字在词汇表中对应的索引
    :param output_labels_file: 输出标签索引文件的完整路径，标签索引文件中的内容是输入语料中每个字对应的分词标签编号，采用SBIE标签，对应编号为0,1,2,3
    :param dict_file: 词典文件的完整路径
    :param input_dict: 指定是否输入词典，若为True，则使用dict_file指定的词典，若为False，则根据语料和vocab_size生成词典，并输出至dict_file指定的位置，默认为False
    :param output_raw_file: 指定是否输出语料库未切分的原始文件，默认为False
    :param raw_file: 输出的语料库未切分的原始语料文件完整路径
    """
    self.input_file = input_file
    self.output_words_file = output_words_file
    self.output_labels_file = output_labels_file
    self.dict_file = dict_file
    self.input_dict = input_dict
    self.vocab_size = vocab_size  # 词汇表大小
    # 指示是否输出原始文件
    if raw_file == None or raw_file == '':
      self.output_raw_file = False
    else:
      self.output_raw_file = True
      self.raw_file = raw_file  # 输出的原始文件名
    self.vocab_count = 0  # 语料库中字符数量，只在自动生成词汇表时会设置
    self.SPLIT_CHAR = '  '  # 分隔符：双空格
    self.sentences = self.read_sentences()  # 从输入文件中读取的句子列表
    self.words_index = []  # 语料文件中每个字对应的索引，以句子为单位
    self.labels_index = []  # 语料库中每个字对应的索引，采用SBIE标记，以句子为单位
    self.count = [['UNK', 0], ['STRT', 0],
                  ['END', 0]]  # 字符数量，其中'UNK'表示词汇表外的字符，'STAT'表示句子首字符之前的字符，'END'表示句子尾字符后面的字符，这两字符用于生成字的上下文
    # 根据是否指定词典路径来初始化词典，若指定，使用给定词典，未指定，根据语料生成
    # 词典中项表示字符编号，从0开始，{'UNK':0,'STRT':1,'END':2,'我':3,'们':4}
    if self.input_dict:
      self.dictionary = self.read_dictionary(self.dict_file)
    else:
      self.dictionary = self.build_dictionary()

  def read_sentences(self):
    file = open(self.input_file, 'r', encoding='utf-8')
    content = file.read()
    sentences = re.sub('[ ]+', self.SPLIT_CHAR, strQ2B(content)).splitlines()  # 将词分隔符统一为双空格
    sentences = list(filter(None, sentences))  # 去除空行
    file.close()
    return sentences

  def build_raw_corpus(self):
    file = open(self.raw_file,'w',encoding='utf-8')
    for sentence in self.sentences:
      file.write(sentence.replace(' ','')+'\n')
    file.close()

  def build_dictionary(self):
    dictionary = {}
    words = ''.join(self.sentences).replace(' ', '')
    self.vocab_count = len(collections.Counter(words))
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - 3))

    for word, _ in self.count:
      dictionary[word] = len(dictionary)
    return dictionary

  def read_dictionary(self, dict_path):
    dict_file = open(dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = dict_item[1]
    dict_file.close()
    return dictionary

  def build_basic_dataset(self):
    unk_count = 0
    # 给语料中的每个字标对应的序号
    for sentence in self.sentences:
      sentence = sentence.replace(' ', '')
      sen_data = []
      for word in sentence:
        if word in self.dictionary:
          index = self.dictionary[word]
        else:
          index = 0
          unk_count += 1
        sen_data.append(index)
      self.words_index.append(sen_data)
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
  def build_test_corpus(self,filename):
    with open(filename,'w',encoding='utf-8') as file:
      for _,(sentence,sentence_label) in enumerate(zip(self.sentences,self.labels_index)):
        file.write(sentence.replace(' ','')+'\n')
        file.write(' '.join(map(lambda i:str(i),sentence_label))+'\n')

  def build_exec(self):
    self.build_basic_dataset()
    self.build_corpus_dataset()
    words_file = open(self.output_words_file, 'w+', encoding='utf-8')
    labels_file = open(self.output_labels_file, 'w+', encoding='utf-8')

    for _, (words, labels) in enumerate(zip(self.words_index, self.labels_index)):
      words_file.write(' '.join(str(word) for word in words) + '\n')
      labels_file.write(' '.join(str(label) for label in labels) + '\n')
    if not self.input_dict:
      dict_file = open(self.dict_file, 'w+', encoding='utf-8')
      for (word, index) in self.dictionary.items():
        dict_file.write(word + ' ' + str(index) + '\n')
      dict_file.close()
    words_file.close()
    labels_file.close()
    if self.output_raw_file:
      self.build_raw_corpus()
    self.build_test_corpus('data/test.utf8')


if __name__ == '__main__':
  prepare_pku = PrepareData(constant.VOCAB_SIZE, 'corpus/pku_training.utf8', 'corpus/pku_training_words.txt',
                            'corpus/pku_training_labels.txt', 'corpus/pku_training_dict.txt','corpus/pku_training_raw.utf8')
  prepare_pku.build_exec()
  # prepare_msr = PrepareData(constant.VOCAB_SIZE,'corpus/msr_training.utf8', 'corpus/msr_training_words.txt',
  #                           'corpus/msr_training_labels.txt', 'corpus/msr_training_dict.txt')
  # prepare_msr.build_exec()
