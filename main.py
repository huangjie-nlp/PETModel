# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 23:14
@Auth ： jiesus
@File ：framework.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""

from framework.framework import Framework
from config.config import Config
import torch
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
config = Config()

framework = Framework(config)
framework.train()
