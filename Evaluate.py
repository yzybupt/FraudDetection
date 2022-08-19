import numpy as np
from sklearn.metrics import roc_auc_score

"""Data structure for storing results information"""
class Results():
    def __init__(self):
        self.AUC = 0
        self.precision = 0
        self.BAC = 0
        self.NDCG = 0
        self.sensitivity = 0


def evaluate(y_prob, y_true, cut_off):
    results = Results()
    results.AUC = roc_auc_score(y_true, y_prob[:,1]) # the second column of y_prob is prob of label '1'


    """Calculate the precision, balanced accuracy by top 1% as threshold"""
    k = round(len(y_true) * cut_off)
    temp = y_prob[:,1].copy()
    list_y_prob = list(y_prob[:,1])
    list_y_prob.sort(reverse=True)
    threshold = list_y_prob[k - 1]
    print(threshold)
    print(k)

    num = 0
    for i in range(len(temp)):
        if temp[i] >= threshold and num <= k:
            num = num + 1
            temp[i] = 1
        else:
            temp[i] = 0


    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(temp)):
        if temp[i] == 1 and y_true[i] == 1:
            tp = tp + 1
        elif temp[i] == 1 and y_true[i] == 0:
            fp = fp + 1
        elif temp[i] == 0 and y_true[i] == 1:
            fn = fn + 1
        else:
            tn = tn + 1
    results.precision = tp / (fp + tp)
    results.sensitivity = tp / (tp + fn)
    results.BAC = 0.5 * (tp / (tp + fn) + tn / (tn + fp))

    """Calculate the NDCG@K where k = 0.01"""
    IDCG = 0
    for i in range(k):
        IDCG = IDCG + (1 / np.log2(i + 1 + 1))

    DCG = 0
    index = 1
    for i in range(len(temp)):
        if temp[i] == 1:
            if y_true[i] == 1:
                DCG = DCG + (1 / np.log2(index + 1))
            index = index + 1
    results.NDCG = DCG / IDCG

    assert index - 1 == k

    return results