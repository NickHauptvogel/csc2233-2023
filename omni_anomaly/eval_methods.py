# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    fdr = TP / (TP + FN)
    far = FP / (FP + TN)
    return f1, fdr, far, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, min):
    """
    Find all scores by using all scores as threshold.

    Returns:
        array: array for results
    """
    result_array = []
    thresholds = list(set([int(i) for i in score if i > min]))
    thresholds.sort()
    for threshold in tqdm(thresholds):
        target = calc_seq(score, label, threshold, calc_latency=True)
        target.append(threshold)
        result_array.append(target)
    result_array = np.array(result_array)
    return result_array


def bf_search_binary(score, label, min):
    # Search for threshold with FAR = 0.01 with binary search
    thresholds = list(set([int(i) for i in score if i > min]))
    thresholds.sort()
    max, min = thresholds[-1], thresholds[0]
    while max - min > 1:
        mid = int((max + min) / 2)
        target = calc_seq(score, label, mid)
        if target[2] > 0.01:
            max = mid
        else:
            min = mid

    target = calc_seq(score, label, min)
    print("Threshold: ", min, " Result: ", target)
