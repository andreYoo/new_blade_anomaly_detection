#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : make_data.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 20:53:30
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
import shutil
import random
import argparse
import glob
import glob
from xml.etree.ElementTree import parse
True_path =  'C:/Users/Schmizt/Desktop/new_training/'
PATH = 'C:/Users/Schmizt/Desktop/new2_training'
list_xmls = glob.glob(PATH)
dataset = './data/dataset/bfd_dataset_v2.txt'
dataset_v2 = './data/dataset/bfd_dataset_vtmp.txt'

# with open(dataset,'w') as wf:
#     for _xml in list_xmls:
#         tree = parse(_xml)
#         root = tree.getroot()
#         _obj_list = root.findall("object")
#         _name = root.findtext("path")
#         _save_str = _name.split("\\")[-1]
#         _save_str = True_path+_save_str
#         for _obj  in _obj_list:
#             _cls = _obj.find("name").text
#             if _cls!='1':
#                 _cls = '1'
#             _xmin = _obj.find("bndbox").findtext("xmin")
#             _ymin = _obj.find("bndbox").findtext("ymin")
#             _xmax = _obj.find("bndbox").findtext("xmax")
#             _ymax = _obj.find("bndbox").findtext("ymax")
#             _str = _xmin+','+_ymin+','+_xmax+','+_ymax+','+_cls
#             _save_str = _save_str + ' '+_str
#         print(_save_str)
#         wf.writelines(_save_str+'\n')
# wf.close()


def gen_dataset_txt(dataset_path,txt_file_path):
    _t_dataset_path = os.path.join(dataset_path,'*.xml')
    list_xmls = glob.glob(_t_dataset_path)
    with open(txt_file_path, 'w') as wf:
        for _xml in list_xmls:
            tree = parse(_xml)
            root = tree.getroot()
            _obj_list = root.findall("object")
            _name = root.findtext("path")
            _save_str = _name.split("\\")[-1]
            _save_str = os.path.join(dataset_path,_save_str)
            for _obj in _obj_list:
                _cls = _obj.find("name").text
                if _cls != '1':
                    _cls = '1'
                _xmin = _obj.find("bndbox").findtext("xmin")
                _ymin = _obj.find("bndbox").findtext("ymin")
                _xmax = _obj.find("bndbox").findtext("xmax")
                _ymax = _obj.find("bndbox").findtext("ymax")
                _str = _xmin + ',' + _ymin + ',' + _xmax + ',' + _ymax + ',' + _cls
                _save_str = _save_str + ' ' + _str
            print(_save_str)
            wf.writelines(_save_str + '\n')
    wf.close()
    return


if __name__ == '__main__':
    gen_dataset_txt(PATH,'./tmp_results.txt')
