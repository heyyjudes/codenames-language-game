
import numpy as np
from pyemd import emd
import math
from scipy.stats import entropy
import warnings

def check_hist(hist):
    try:
        assert 0.99 < sum(hist) < 1.01
        assert min(hist) >= 0
    except:
        print("hist error")
        print(hist)

def wasserstein(hist1, hist2):
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)
    assert len(hist1) == len(hist2)
    check_hist(hist1)
    check_hist(hist2)

    N = len(hist1)
    dist_mat = np.ones((N, N))
    return emd(hist1, hist2, dist_mat)

def jsd(hist1, hist2):
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)
    assert len(hist1) == len(hist2)
    check_hist(hist1)
    check_hist(hist2)

    hist_avg = 0.5 * (hist1 + hist2)
    return 0.5 * (entropy(hist1, hist_avg) + entropy(hist2, hist_avg))

def top_elements(array, k):
    # TODO: modifiy for deterministic behaviour in case of tied model
    arr = np.array(array)
    ind = np.argpartition(arr, -k)[-k:]
    return ind[np.argsort(arr[ind])][::-1]

def topk_dist(top_mturk, model_dist):
    top_model = top_elements(model_dist, len(model_dist))
    return top_model.tolist().index(top_mturk)

def topk_acc(topk_list, k):
    if type(topk_list[0]) == list:
        topk_flat = [sub for l in topk_list for sub in l]
    else:
        topk_flat = topk_list
    acc_arr = [1.0 if x <= k-1 else 0.0 for x in topk_flat if x not in [None, np.nan]]
    acc = np.mean(acc_arr)
    print(len(acc_arr))
    return round(acc, 3), std_err_from_mean(acc, len(topk_flat))


def topk_avg_acc(topk_list, k):
    acc_arr = []
    for scen in topk_list:
        result = [1.0 if x <= k - 1 else 0.0 for x in scen if x not in [None, np.nan]]
        if len(result) > 1:
            acc = np.mean([1.0 if x <= k - 1 else 0.0 for x in scen if x not in [None, np.nan]])
            acc_arr.append(acc)
        elif len(result) == 1:
            acc = result[0]
            acc_arr.append(acc)
    acc = np.mean(acc_arr)
    if len(result) > 1:
        print(len(acc_arr))
        print(len(result))
        return round(acc, 3), np.std(acc_arr)/len(acc_arr)
    else:
        print(len(acc_arr))
        return round(acc, 3), std_err_from_mean(acc, len(acc_arr))


def topk_acc_discard(topk_list, k):
    if type(topk_list[0]) == list:
        incorrect_arr = np.load('listener_incorrect.npy')
        topk_list = topk_list + incorrect_arr
        topk_flat = [sub for l in topk_list for sub in l]
    else:
        for i in range(len(topk_list)):
            if i in discard_arr:
                topk_list[i] = np.nan
        topk_flat = topk_list
    acc = np.mean([1.0 if x <= k-1 else 0.0 for x in topk_flat if x not in [None, np.nan]])
    return round(acc * 100, 1)

def std_err_from_mean(mu, n):
    p = mu
    SE = math.sqrt(p*(1-p)/n)
    return SE