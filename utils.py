# -*- coding:UTF-8 -*-
import os
import re


def strQ2B(ustring):
  '''全角转半角'''
  rstring = ''
  for uchar in ustring:
    inside_code = ord(uchar)
    if inside_code == 12288:  # 全角空格直接转换
      inside_code = 32
    elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
      inside_code -= 65248
    rstring += chr(inside_code)
  return rstring


def escape(text):
  '''html转义'''
  text = (text.replace("&quot;", "\"").replace("&ldquo;", "“").replace("&rdquo;", "”")
          .replace("&middot;", "·").replace("&#8217;", "’").replace("&#8220;", "“")
          .replace("&#8221;", "\”").replace("&#8212;", "——").replace("&hellip;", "…")
          .replace("&#8226;", "·").replace("&#40;", "(").replace("&#41;", ")")
          .replace("&#183;", "·").replace("&amp;", "&").replace("&bull;", "·")
          .replace("&lt;", "<").replace("&#60;", "<").replace("&gt;", ">")
          .replace("&#62;", ">").replace("&nbsp;", " ").replace("&#160;", " ")
          .replace("&tilde;", "~").replace("&mdash;", "—").replace("&copy;", "@")
          .replace("&#169;", "@").replace("♂", "").replace("\r\n|\r", "\n").replace('&nbsp', ' '))
  return text


def read_sogou_report():
  base = 'Reduced/'
  types = os.listdir(base)
  sentences = []
  count = 0
  index = 0
  for type in types:
    # type = 'C000008'
    docs = os.listdir(base + type)
    for doc in docs:
      file = None
      try:
        file = open(base + type + '/' + doc, 'r', encoding='gbk')
        content = escape(strQ2B(file.read())).replace(r'\s', '').replace(r'\n\d+\n', '')
        lines = re.split(r'\n', re.sub(r'[ \t\f]+', r'', content))
        for line in lines:
          sentences.extend(line.split('。'))
        # break
        file.close()
      except UnicodeDecodeError as e:
        count += 1
        file.close()
        # sentences.append(content)

  return sentences

def estimate_cws(current_labels,correct_labels):
  cor_dict = {}
  curt_dict = {}
  curt_start = 0
  cor_start = 0
  for label_index,(curt_label,cor_label) in enumerate(zip(current_labels,correct_labels)):
    if cor_label == 0:
      cor_dict[label_index] = label_index + 1
    elif cor_label == 1:
      cor_start = label_index
    elif cor_label == 3:
      cor_dict[cor_start] = label_index + 1

    if curt_label == 0:
      curt_dict[label_index] = label_index + 1
    elif curt_label == 1:
      curt_start = label_index
    elif curt_label == 3:
      curt_dict[curt_start] = label_index + 1

  cor_count = 0
  recall_length = len(curt_dict)
  prec_length = len(cor_dict)
  for curt_start in curt_dict.keys():
    if curt_start in cor_dict and curt_dict[curt_start] == cor_dict[curt_start]:
      cor_count += 1

  return  cor_count,prec_length,recall_length

if __name__ == '__main__':
  sentences = read_sogou_report()
  file = open('corpus/sougou.txt', 'w', encoding='utf-8')
  print(len(sentences))
  content = ''.join(sentences)
  content = re.sub('[\0]', '', content)
  file.write(content)
  file.close()
