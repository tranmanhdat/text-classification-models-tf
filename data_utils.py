import os

import wget
import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

def read_text(path_file):
    X, y, labels= [], [], []
    len_X = []
    with open(path_file, "r", encoding="UTF-8") as f_read:
        for line in f_read:
            elements = line.split("\t")
            if 100 < len(elements[1]):
                X.append(elements[1].rstrip())
                len_X.append(len(elements[1]))
                y.append(elements[0])
                if elements[0] not in labels:
                    labels.append(elements[0])
    print("{:.2f}".format(sum(len_X)/len(len_X)))
    print("{}".format(max(len_X)))
    print("{}".format(min(len_X)))
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    return X, y, labels

def build_char_dataset(path_file, document_max_len):
    alphabet = "ngoàirabêcũsẽhỗtợvệxâydựươìđạkáọpòuếlĩảmậổộeíẫờwf1ôỉầấụặóịởữềăốq3ớẩứửồắú5ýằ92ẵ4õ6ễừ07ùẻẹể​ủjãzỹ8ỏéẳèỷỳỡỵ-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    # alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    alphabet = ''.join(sorted(alphabet))
    # print(alphabet)
    X, y, labels = read_text(path_file)
    # for i in range(0,5):
    #     # print(X[i])
    #     print(y[i])
    le = LabelEncoder()
    if os.path.isfile("classes.npy"):
        le.classes_ = np.load("classes.npy")
    else:
        le.fit(labels)
        np.save('classes.npy', le.classes_)
    y = le.fit_transform(y)
    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)
    alphabet_size = len(alphabet) + 2

    x = list( map( lambda content: list( map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), X) )
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(
        map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]],
            x))
    # for i in range(0,5):
    #     print(x[i])
    #     print(y[i])
    return x, y, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
