# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 22:40
@Auth ： jiesus
@File ：models.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""

import torch.nn as nn
import torch
from transformers import BertTokenizer, BertForMaskedLM

class PETModel(nn.Module):
    def __init__(self, config):
        super(PETModel, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertForMaskedLM.from_pretrained("bert-base-chinese")
        # 这一步目的是为了让模板里的[MASK]预测范围控制在bert vocab中的[差， 好]
        self.label_list = [self.tokenizer.convert_tokens_to_ids("差"), self.tokenizer.convert_tokens_to_ids("好")]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        input_ids = data["input_ids"].to(self.device)
        mask = data["mask"].to(self.device)
        logits = self.model(input_ids, attention_mask=mask).logits
        lm_logits = logits[:, self.config.mask_idx]
        lm_logits = lm_logits[:, self.label_list]
        return lm_logits