# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 23:14
@Auth ： jiesus
@File ：framework.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""
import torch
import json
from models.models import PETModel
from dataloader.dataloader import MyDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

class Framework():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[0]

    def train(self):

        train_dataset = MyDataset(self.config, self.config.train_fn)
        dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                collate_fn=collate_fn, pin_memory=True)

        dev_dataset = MyDataset(self.config, self.config.dev_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn, pin_memory=True)

        model = PETModel(self.config).to(self.device)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epoch):
            epoch_loss = 0
            for data in tqdm(dataloader):
                logits = model(data)
                optimizer.zero_grad()
                loss = loss_func(logits, data["target"].to(self.device))
                loss.backward()
                epoch_loss += loss.item()
            print("loss: {:5.4f}".format(epoch_loss))
            self.evaluate(model, dev_dataloader)

    def evaluate(self, model, dataloader):
        model.eval()
        predict, grundtrue = [], []
        with torch.no_grad():
            for data in dataloader:
                logits = model(data)
                pred = torch.argmax(logits, dim=1)
                for i in pred.cpu().tolist():
                    predict.append(self.id2label[str(i)])
                grundtrue.extend(list(data["label"]))
        print(classification_report(np.array(grundtrue), np.array(predict)))
        model.train()

