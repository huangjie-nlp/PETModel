# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 22:52
@Auth ： jiesus
@File ：dataloader.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from transformers import BertTokenizer
import numpy as np

class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config
        self.df = pd.read_csv(fn)
        self.sentence = self.df.sentence.tolist()
        self.label = self.df.label.tolist()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)[1]

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        label = self.label[idx]
        label_id = self.label2id[label]
        sentence = self.sentence[idx]
        token = ["[CLS]"] + self.tokenizer.tokenize(sentence)[:self.config.max_len] + ["[SEP]"]
        token_len = len(token)
        token2id = self.tokenizer.convert_tokens_to_ids(token)

        mask = [1] * token_len
        mask = np.array(mask)
        input_ids = np.array(token2id)

        return sentence, label, label_id, input_ids, mask, token_len

def collate_fn(batch):
    sentence, label, label_id, input_ids, mask, token_len = zip(*batch)

    cur_batch = len(batch)
    text_max_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, text_max_len).zero_()
    batch_mask = torch.LongTensor(cur_batch, text_max_len).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))

    return {"sentence": sentence,
            "label": label,
            "input_ids": batch_input_ids,
            "mask": batch_mask,
            "target": torch.tensor(label_id, dtype=torch.long)}
