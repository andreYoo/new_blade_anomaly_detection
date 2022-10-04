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

PATH = 'C:/Users/Schmizt/Desktop/dataset/*.png'
list_png = glob.glob(PATH)
dataset = './data/dataset/bfd_dataset_test.txt'

with open(dataset,'w') as wf:
    for _f in list_png:
        _save_str =  _f
        print(_save_str)
        wf.writelines(_save_str+'\n')
wf.close()

