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


if __name__ == '__main__':
  sentences = read_sogou_report()
  file = open('corpus/sougou.txt', 'w', encoding='utf-8')
  print(len(sentences))
  content = ''.join(sentences)
  content = re.sub('[\0]', '', content)
  file.write(content)
  file.close()
