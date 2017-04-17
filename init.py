#-*- coding: UTF-8 -*-
from prepare_data import PrepareData
from transform_data_dnn import TransformDataDNN
import constant
from shutil import copyfile

def init():
  prepare_pku = PrepareData(constant.VOCAB_SIZE, 'corpus/pku_training.utf8', 'corpus/pku_training_words.txt',
                            'corpus/pku_training_labels.txt', 'corpus/pku_training_dict.txt')
  prepare_pku.build_exec()
  dict_name = 'corpus/pku_training_dict.txt'
  copyfile(dict_name,'corpus/dict.utf8')
  trans_dnn = TransformDataDNN(constant.DNN_SKIP_WINDOW, True)
  trans_dnn.generate_exe()

if __name__ == '__main__':
  init()