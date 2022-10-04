#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = "C:/Users/Schmizt/Desktop/powerINS_project/pwins_new/blade-fault-detector/detector/YOLOV3/data/classes/yymnist.names"
__C.YOLO.ANCHORS              = "C:/Users/Schmizt/Desktop/powerINS_project/pwins_new/blade-fault-detector/detector/YOLOV3/data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./tmp_results.txt"
__C.TRAIN.BATCH_SIZE          = 12
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = False
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 1
__C.TRAIN.EPOCHS              = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./tmp_results.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 544
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "C:/Users/Schmizt/Desktop/powerINS_project/pwins_new/blade-fault-detector/detector/YOLOV3/data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.1
__C.TEST.IOU_THRESHOLD        = 0.2


