import collections
from utils import strQ2B

SPLIT_CHAR = '  '

def build_dataset_from_raw(sentences,vocab_size):
  words = ''.join(sentences).replace(' ', '')
  #print(words)
  count = [['UNK', -1]] # 字符数量
  print(len(words))
  print(len(collections.Counter(words)))
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
    data.extend(senData)
  count[0][1] = unk_count
  # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary #, reverse_dictionary


def build_dataset_from_annotated(sentences, vocab_size, split_char):
  vocab_index, count, dictionary = build_dataset_from_raw(sentences, vocab_size)
  label_index = list()
  index = 0
  empty = 0
  for sentence in sentences:
    words = sentence.strip().split(split_char)
    for word in words:
      #print(word)
      l = len(word)
      if l == 0:
        empty += 1
        #continue
      elif l == 1:
        if word == ' ':
          print('a')
        label_index.append(0)
      else:
        label_index.append(1)
        label_index.extend([2] * (l - 2))
        label_index.append(3)
      index += l

  print(empty)
  return vocab_index, label_index, count, dictionary


if __name__ == '__main__':
  vocab_size = 1000
  sentences = strQ2B(open('pku_training.utf8',encoding='utf-8').read()).splitlines()[10038:10039]
  # sentences.extend(open('msr_traning.utf8', encoding='utf-8').read().splitlines())
  vocab_index, label_index, count, dictionary = build_dataset_from_annotated(sentences, vocab_size, SPLIT_CHAR)
  print(len(vocab_index),len(label_index))