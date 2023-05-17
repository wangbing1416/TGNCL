from math import log, pow
import torch
import numpy as np
# torch.cuda.set_per_process_memory_fraction(0.5, 0)


def get_class_weight(label, A, B):
    num_label = [0 for x in range(len(label[0]))]
    for doc in label:
        num_label = [num_label[x]+doc[x] for x in range(len(num_label))]
    N = len(label)
    C = (log(N)-1) * pow((B + 1), A)
    weight = [1 + C * pow(num_label[x] + B, A * -1) for x in range(len(label[0]))]

    return torch.tensor(weight)


def get_pmi(labels: list):
    co_matrix = np.zeros([len(labels[0]), len(labels[0])])
    for label in labels:
        for i in range(len(label)):
            if label[i] != 0:
                for j in range(i + 1, len(label)):
                    if label[j] != 0:
                        co_matrix[i, j] = co_matrix[i, j] + 1
                        co_matrix[j, i] = co_matrix[j, i] + 1
    num = np.sum(co_matrix)
    label_prob = np.sum(co_matrix, axis=0) / num
    co_matrix = co_matrix / num * 2
    for i in range(len(labels[0])):
        for j in range(i + 1, len(labels[0])):
            if co_matrix[i][j] > 0 and co_matrix[i][j] > 0:
                co_matrix[i][j] = np.log2(co_matrix[i][j] / (label_prob[i] * label_prob[j]))
                co_matrix[j][i] = np.log2(co_matrix[j][i] / (label_prob[j] * label_prob[i]))
            if co_matrix[i][j] <= 0:
                co_matrix[i][j] = 0
                co_matrix[j][i] = 0

    return co_matrix.astype(np.float32)


def topk(array, k):
    topk_index = []
    a = array.copy()
    for i in range(k):
        w = np.argmax(a)
        topk_index.append(w)
        a[w] = 0
    return topk_index



