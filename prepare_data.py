import collections
import os
import re
from functools import reduce
from utils import escape,strQ2B

SPLIT_CHAR = '  '

def build_dataset_from_raw(sentences,vocab_size):
  words = ''.join(sentences).replace(' ', '')
  count = [['UNK', -1]] # 字符数量
  count.extend(collections.Counter(words).most_common(vocab_size - 1))
  dictionary = dict() # 字符编号，从0开始，{'我':0,'们':1}
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  # 给语料中的每个字标对应的序号
  for sentence in sentences:
    sentence = sentence.replace(' ','')
    senData = list()
    for word in sentence:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0
        unk_count += 1
      senData.append(index)
    data.append(senData)
  count[0][1] = unk_count
  # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary #, reverse_dictionary


def build_dataset_from_annotated(sentences, vocab_size, split_char):
  vocab_index, count, dictionary = build_dataset_from_raw(sentences, vocab_size)
  label_index = list()
  empty = 0
  for sentence in sentences:
    sentence_label = list()
    words = sentence.strip().split(split_char)
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
      #index += l
    label_index.append(sentence_label)

  return vocab_index, label_index, count, dictionary


def read_sogou_report():
  base = 'Reduced/'
  types = os.listdir(base)
  sentences = []
  count = 0
  index = 0
  #for type in types:
  type='C000008'
  docs = os.listdir(base + type)
  for doc in docs:
    try:
      file = open(base + type + '/' + doc, 'r',encoding='gbk')
      content = escape(strQ2B(file.read())).replace(r'\s','').replace(r'\n\d+\n','')
      # if index == 0:
      lines = re.split(r'\n',re.sub(r'[ \t\f]+',r'',content))
      for line in lines:
        sentences.extend(line.split('。'))
      #  break
      file.close()
    except UnicodeDecodeError as e:
      count += 1
      file.close()
      # sentences.append(content)

  return sentences


def read_raw_data ():
  stopList = open('StopList.txt').read().splitlines()
  stopList.append('\n')

  sentences = read_sogou_report()
  file = open('sentences.txt', 'w', encoding='utf-8')
  for sentence in sentences:
    if len(sentence) > 0 and sentence not in stopList:
      file.write(sentence + '\n')


def read_train_data (vocab_size):
  # vocab_size = 1000
  pku = re.sub('[ ]+', SPLIT_CHAR, strQ2B(open('pku_training.utf8', encoding='utf-8').read()))
  # sentences.extend(open('msr_traning.utf8', encoding='utf-8').read().splitlines())
  sentences = pku.splitlines()
  # vocab_index, label_index, count, dictionary = build_dataset_from_annotated(sentences, vocab_size, SPLIT_CHAR)
  return tuple(sentences), build_dataset_from_annotated(sentences, vocab_size, SPLIT_CHAR)


if __name__ == '__main__':
  vocab_size = 1000
  pku = re.sub('[ ]+',SPLIT_CHAR, strQ2B(open('pku_training.utf8',encoding='utf-8').read()))#.splitlines()[10038:10039])
  # sentences.extend(open('msr_traning.utf8', encoding='utf-8').read().splitlines())
  sentences = pku.splitlines()
  vocab_index, label_index, count, dictionary = build_dataset_from_annotated(sentences, vocab_size, SPLIT_CHAR)
  v = reduce(lambda x,y :x+y ,vocab_index)
  l = reduce(lambda x,y :x+y ,label_index)
  print(len(v), len(l))

  # print(len(vocab_index),len(label_index))