import os
import re
from utils import escape,strQ2B

def readSogouReport():
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

  #print(len(reports))
  #print(count)
  return sentences



if __name__ == '__main__':
  stopList = open('StopList.txt').read().splitlines()
  stopList.append('\n')
  # print(stopList)
  sentences = readSogouReport()
  file = open('sentences.txt','w',encoding='utf-8')
  for sentence in sentences:
    if len(sentence) > 0 and sentence not in stopList:
      file.write(sentence+'\n')

