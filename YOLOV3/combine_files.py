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


file_list = ['./data/dataset/bfd_dataset.txt','./data/dataset/bfd_dataset_v2.txt']
total_file_path = './data/dataset/true_tf_dataset.txt'
with open(total_file_path, 'w') as wf:
    for _pth in file_list:
        with open(_pth, 'r') as rf:
            while True:
                line = rf.readline()
                if not line: break
                else:
                    wf.writelines(line)
        rf.close()
wf.close()

