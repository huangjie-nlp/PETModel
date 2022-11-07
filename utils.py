# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/7 22:32
@Auth ： jiesus
@File ：utils.py
@IDE ：PyCharm
@e-mail: 728155808@qq.com
"""
import pandas as pd
import json

def processing(file):
    sentence, label = [], []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        s, l = line.strip("\n").split("\t")
        sentence.append(s)
        if l == '1':
            t = "好"
        elif l == '0':
            t = "差"
        else:
            raise
        label.append(t)
    return sentence, label

if __name__ == '__main__':
    file = "src_data/train.tsv"
    sentence, label = processing(file)
    pd.DataFrame({"sentence": sentence, "label": label}).to_csv("dataset/train.csv", index=False)
    # label = [{0: "差", 1: "好"}, {"差": 0, "好": 1}]
    # json.dump(label, open("dataset/schema.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)