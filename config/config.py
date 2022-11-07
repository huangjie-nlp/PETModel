# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 22:48
@Auth ： jiesus
@File ：config.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""

class Config():
    def __init__(self):
        self.template = "这家宾馆很[MASK]，" # 在[MASK]这个位置预测 差 or 好
        self.mask_idx = 6
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.train_fn = "dataset/train.csv"
        self.dev_fn = "dataset/dev.csv"
        self.schema_fn = "dataset/schema.json"
        self.max_len = 256
        self.epoch = 3