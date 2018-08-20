import pickle
import rsa_model
import data_utils
import numpy as np
import scipy.stats
import heapq
import itertools
import entropy
import matplotlib.pyplot as plt

NUM_ADJ = 3
NUM_NOUN = 3
NUM_NPAIR = 3
HEAP_SIZE = 500
TOTAL_ADJ = 100
TOTAL_NOUN = 50


def speaker_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, agg_func=np.prod, alpha=1.0, idx=0, full_mat=False):
    '''
    P(y | M, D)
    :param adj_features: adj_mat feature
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param agg_func: aggregation function
    :param alpha: pragmatic parameter
    :param idx: index of noun pair
    :param full_mat: whether to return full mat including raw matrix
    :return: rsa matrix
    '''
    raw, index = rsa_model.extract_scenario_table(scenario_adj, scenario_nouns, adj_features, agg_func)
    rsa_mat = rsa_model.rsa_speaker(raw, alpha)

    if full_mat:
        rsa_mat['raw'] = raw
        return rsa_mat
    # assert(index == idx)
    rsa_mat['ls'] = rsa_mat['ls'][:, idx]
    rsa_mat['ll'] = rsa_mat['ll'][:, idx]
    rsa_mat['ps'] = rsa_mat['ps'][:, idx]

    return rsa_mat


def listener_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, idx, agg_func=np.prod, alpha=1.0, full_mat=False):
    '''
    P(y | M, D)
    :param adj_features: adj_mat feature
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param agg_func: aggregation function
    :param alpha: pragmatic parameter
    :param idx: index of adjective
    :return: rsa matrix
    '''
    raw, _ = rsa_model.extract_scenario_table(scenario_adj, scenario_nouns, adj_features, agg_func)
    rsa_mat = rsa_model.rsa_listener(raw, alpha)
    if full_mat:
        rsa_mat['raw'] = raw
        return rsa_mat
    rsa_mat['ll'] = rsa_mat['ll'][idx, :]
    rsa_mat['ls'] = rsa_mat['ls'][idx, :]
    rsa_mat['pl'] = rsa_mat['pl'][idx, :]
    return rsa_mat


def average_across_models(model_list, scenario_adj, scenario_nouns, choice_func, comp='speaker', idx=0):
    '''
    p(y|D)
    :param model_list:  list of models to compare in (adj_mat, prag)
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param choice_func: function with which to evaluate rsa matrices from
    :param comp: speaker or listener
    :param idx: index of selected adjective or noun pair in scenario
    :return: array of prob of output given scenario
    '''
    sum_across_models = []
    # speaker
    if comp == 'speaker':
        for adj_features, prag_setting in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx)
            sum_across_models.append(y_arr[prag_setting])
    # listener
    else:
        for adj_features, prag_setting in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx)
            sum_across_models.append(y_arr[prag_setting])
    sum_across_models = np.asarray(sum_across_models)
    return sum_across_models.mean(axis=0)


def model_prob_given_choice(model_list, scenario_adj, scenario_nouns, choice_func, comp='speaker', idx=0, alpha=1.0):
    '''
    :param model_list:  list of models to compare in (adj_mat, prag)
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param choice_func: function with which to evaluate rsa matrices from
    :param comp: speaker or listener
    :param idx: index of selected adjective or noun pair in scenario
    :return:
    '''
    # P(model|choice)
    prob_model_choice = []
    # speaker
    if comp == 'speaker':
        for (adj_features, prag_setting) in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            prob_model_choice.append(y_arr[prag_setting])
    # listener
    else:
        for (adj_features, prag_setting) in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            prob_model_choice.append(y_arr[prag_setting])
    prob_model_choice = np.asarray(prob_model_choice)
    return prob_model_choice


def specific_utility(model_list, scenario_adj, scenario_nouns, choice_func, comp='speaker', idx=0):
    '''
    specific utility U(y, M)
    :param model_list: list of models to compare in (adj_mat, prag)
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param choice_func:  function with which to evaluate rsa matrices from
    :param comp: speaker or listener
    :param idx: index of selected adjective or noun pair in scenario
    :return: array of utility answer choice
    '''
    # U(y, D)
    sum_across_models = []
    if comp == 'speaker':
        for adj_features, prag_setting in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx)
            sum_across_models.append(y_arr[prag_setting])
    else:
        for adj_features, prag_setting in model_list:
            y_arr = choice_func(adj_features, scenario_adj, scenario_nouns, idx=idx)
            sum_across_models.append(y_arr[prag_setting])
    sum_across_models = np.asarray(sum_across_models)

    # scipy.stats documentation says this does not have to be normalized
    util_arr = []
    x, y = sum_across_models.shape
    for i in range(y):
        uniform_dist = np.zeros((x,))
        uniform_dist.fill(1/x)
        kl = scipy.stats.entropy(sum_across_models[:, i], uniform_dist)
        util_arr.append(kl)
    return np.asarray(util_arr)


def util_total(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp='speaker', scen_perm=False):
    '''
    Total util function for specific index in a scenario or a geometric mean of all scenarios
    :param comp_model_list: list of models to compare in (adj_mat, prag)
    :param scenario_adjs: list of adjections in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param choice_func: function with which to evaluate rsa matricies from
    :param comp: speaker or listener
    :param scen_perm: use all indices per scenario
    :return:  total utility float
    '''
    if comp == 'speaker':
        if scen_perm:
            util_arr = []
            for i in range(NUM_NPAIR):
                p_y_D = average_across_models(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
                U_y_D = specific_utility(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
                util_arr.append(np.dot(p_y_D, U_y_D))
            util = np.prod(np.asarray(util_arr))**(1.0/len(util_arr))

        else:
            i = 0
            p_y_D = average_across_models(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
            U_y_D = specific_utility(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
            util = np.dot(p_y_D, U_y_D)
    elif comp == 'listener':
        if scen_perm:
            util_arr = []
            for i in range(NUM_ADJ):
                p_y_D = average_across_models(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
                U_y_D = specific_utility(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
                util_arr.append(np.dot(p_y_D, U_y_D))
            util = np.prod(np.asarray(util_arr))**(1.0/len(util_arr))
        else:
            i = 0
            p_y_D = average_across_models(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
            U_y_D = specific_utility(comp_model_list, scenario_adjs, scenario_nouns, choice_func, comp, idx=i)
            util = np.dot(p_y_D, U_y_D)
    elif comp == 'all':

        model_list_listener = [(model, prag[0] + 'l') for (model, prag) in comp_model_list]
        if scen_perm:
            util_arr = []
            for i in range(NUM_ADJ):
                p_y_D = average_across_models(model_list_listener, scenario_adjs, scenario_nouns, choice_func[1],
                                              comp='listener', idx=i)
                U_y_D = specific_utility(model_list_listener, scenario_adjs, scenario_nouns, choice_func[1],
                                         comp='listener', idx=i)
                util_arr.append(np.dot(p_y_D, U_y_D))
            try:
                listener_util = np.prod(np.asarray(util_arr))**(1.0/len(util_arr))
            except RuntimeWarning:
                listener_util = 0

            model_list_speaker = [(model, prag[0] + 's') for (model, prag) in comp_model_list]
            util_arr = []
            for i in range(NUM_NPAIR):
                p_y_D = average_across_models(model_list_speaker, scenario_adjs, scenario_nouns, choice_func[0], comp='speaker', idx=i)
                U_y_D = specific_utility(model_list_speaker, scenario_adjs, scenario_nouns, choice_func[0], comp='speaker', idx=i)
                util_arr.append(np.dot(p_y_D, U_y_D))
            try:
                speaker_util = np.prod(np.asarray(util_arr))**(1.0/len(util_arr))
            except RuntimeWarning:
                speaker_util = 0
            util = np.sqrt(speaker_util*listener_util)

        else:
            model_list_listener = [(model, prag[0] + 'l') for (model, prag) in comp_model_list]
            i = 0
            p_y_D = average_across_models(model_list_listener, scenario_adjs, scenario_nouns, choice_func[1], comp='listener', idx=i)
            U_y_D = specific_utility(model_list_listener, scenario_adjs, scenario_nouns, choice_func[1], comp='listener', idx=i)
            listener_util = np.dot(p_y_D, U_y_D)

            model_list_speaker = [(model, prag[0] + 's') for (model, prag) in comp_model_list]
            i = 0
            p_y_D = average_across_models(model_list_speaker, scenario_adjs, scenario_nouns, choice_func[0], comp='speaker', idx=i)
            U_y_D = specific_utility(model_list_speaker, scenario_adjs, scenario_nouns, choice_func[0], comp='speaker', idx=i)
            speaker_util = np.dot(p_y_D, U_y_D)
            util = np.sqrt(speaker_util * listener_util)
    return util


def abs_max_across_models(model_list, scenario_adj, scenario_nouns, comp='speaker', idx=0, alpha=1.0):
    '''
    return max across models
    :param model_list: list of models to compare in (adj_mat, prag)
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param comp: speaker or listener
    :param idx: index of selected adjective or noun pair in scenario
    :param alpha: pragmatic coefficient
    :return: array of argmax of model
    '''
    sum_across_models = []
    for adj_features, prag_setting in model_list:
        if comp == 'speaker':
            y_arr = speaker_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            sum_across_models.append(y_arr[prag_setting])
        else:
            y_arr = listener_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            sum_across_models.append(y_arr[prag_setting])

    sum_across_models = np.asarray(sum_across_models)
    model_max = np.argmax(sum_across_models, axis=1)

    return model_max


def dist_across_models(model_list, scenario_adj, scenario_nouns, comp='speaker', idx=0, alpha=1.0):
    '''
    return model distribution given scenario
    :param model_list: list of models to compare in (adj_mat, prag)
    :param scenario_adj: list of adjectives in str form
    :param scenario_nouns: sounds in scenario in number / index form
    :param comp: speaker or listener
    :param idx: index of selected adjective or noun pair in scenario
    :param alpha: pragmatic coefficient
    :return: model distributions for given scenario
    '''
    sum_across_models = []
    for adj_features, prag_setting in model_list:
        if comp == 'speaker':
            y_arr = speaker_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            sum_across_models.append(y_arr[prag_setting])
        else:
            y_arr = listener_choice_given_M_D(adj_features, scenario_adj, scenario_nouns, idx=idx, alpha=alpha)
            sum_across_models.append(y_arr[prag_setting])

    sum_across_models = np.asarray(sum_across_models)
    return sum_across_models


def get_adj_sample_arr(x_sampled, n_total, prob_var=True, noun_samples=None, model_list=None):
    '''
    helper function to sample adjectives
    :param x_sampled: already sampled
    :param n_total: total number of adjectives
    :param prob_var: whether to consider variance across all features
    :param noun_samples: nouns sampled
    :param model_list: list of models
    :return: sampled adjectives
    '''
    # return next sampled x_i given previously sampled x_i
    sample_arr = np.asarray(range(0, n_total))
    sample_arr = np.delete(sample_arr, x_sampled)
    if prob_var:
        adj_prob_dist = get_adj_var(model_list, noun_samples, x_sampled)
        adj_scen = np.random.choice(sample_arr, size=1, p=adj_prob_dist)[0]
    else:
        adj_scen = np.random.choice(sample_arr, size=1)[0]
    return adj_scen


def get_noun_sample_arr(x_sampled, n_total, prob_var=True, adj_samples=None, model_list=None):
    '''
    helper function to sample nouns
    :param x_sampled: already sampled
    :param n_total: total number of adjectives
    :param prob_var: whether to consider variance across all features
    :param adj_samples: adjectives sampled
    :param model_list: list of models
    :return: sampled adjectives
    '''
    # return next sampled x_i given previously sampled x_i
    sample_arr = np.asarray(range(0, n_total))
    sample_arr = np.delete(sample_arr, x_sampled)
    if prob_var:
        noun_prob_dist = get_noun_var(model_list, adj_samples, x_sampled)
        noun_scen = np.random.choice(sample_arr, size=1, p=noun_prob_dist)[0]
    else:
        noun_scen = np.random.choice(sample_arr, size=1)[0]
    return noun_scen


def speaker_model_corr_mat(model_comp_list, code_scenarios_speaker, adj_scenarios_speaker):
    '''
    compute correlation of model distributions across all scenarios
    :param model_comp_list: list of models to compare in (adj_mat, prag)
    :param code_scenarios_speaker: codename scenarios
    :param adj_scenarios_speaker: adj scenarios
    :return:return top match percentage and average rank corr matrix
    '''
    model_comp_mat = np.zeros((len(model_comp_list), len(model_comp_list), len(adj_scenarios_speaker)))
    model_comp_abs = np.zeros((len(model_comp_list), len(model_comp_list)))
    for i, scen in enumerate(adj_scenarios_speaker):
        adj_set = adj_scenarios_speaker[i]
        noun_set = code_scenarios_speaker[i]
        model_choices = abs_max_across_models(model_comp_list, adj_set, noun_set,
                                                               comp='speaker')
        model_dist = dist_across_models(model_comp_list, adj_set, noun_set, comp='speaker')

        for j in range(len(model_comp_list)):
            for k in range(j, len(model_comp_list)):
                model_comp_mat[j, k, i] = scipy.stats.spearmanr(model_dist[j], model_dist[k])[0]
                model_comp_mat[k, j, i] = scipy.stats.spearmanr(model_dist[j], model_dist[k])[0]
                if model_choices[j] == model_choices[k] and k > j:
                    model_comp_abs[j, k] += 1
                    model_comp_abs[k, j] += 1
            model_comp_abs[j, j] = len(adj_scenarios_speaker)

    model_comp_mat = np.mean(model_comp_mat, axis=2)
    model_comp_abs /= len(adj_scenarios_speaker)
    return model_comp_abs, model_comp_mat


def listener_model_corr_mat(model_comp_list, code_scenarios_listener, adj_scenarios_listener):
    '''
    compute correlation of model distributions across all scenarios
    :param model_comp_list: list of models to compare in (adj_mat, prag)
    :param code_scenarios_listener: codename scenarios
    :param adj_scenarios_listener: adj scenarios
    :return:return top match percentage and average rank corr matrix
    '''
    model_comp_mat = np.zeros((len(model_comp_list), len(model_comp_list), len(adj_scenarios_listener)))
    model_comp_abs = np.zeros((len(model_comp_list), len(model_comp_list)))

    for i, scen in enumerate(adj_scenarios_listener):
        adj_set = adj_scenarios_listener[i]
        noun_set = code_scenarios_listener[i]
        model_choices = abs_max_across_models(model_comp_list, adj_set, noun_set,
                                                               comp='listener')
        model_dist = dist_across_models(model_comp_list, adj_set, noun_set, comp='listener')

        for j in range(len(model_comp_list)):
            for k in range(j, len(model_comp_list)):
                try:
                    model_comp_mat[j, k, i] = scipy.stats.spearmanr(model_dist[j], model_dist[k])[0]
                    model_comp_mat[k, j, i] = scipy.stats.spearmanr(model_dist[j], model_dist[k])[0]
                except:
                    model_comp_mat[j, k, i] = 0
                    model_comp_mat[j, k, i] = 0
                if model_choices[j] == model_choices[k] and k > j:
                    model_comp_abs[j, k] += 1
                    model_comp_abs[k, j] += 1
            model_comp_abs[j, j] = len(adj_scenarios_listener)

    model_comp_mat = np.mean(model_comp_mat, axis=2)
    model_comp_abs /= len(adj_scenarios_listener)
    return model_comp_abs, model_comp_mat


def get_noun_var(feat_list, adj_arr, delete_arr=None):
    '''
    helper function to compute average relatedness variance for nouns
    :param feat_list: list of models
    :param adj_arr: list of adjectives
    :param delete_arr: section of words to remove
    :return: mean variance
    '''
    mean_noun_prob_dist = np.zeros((len(feat_list), TOTAL_NOUN))
    for j, (feat, _) in enumerate(feat_list):
        noun_prob_dist = np.zeros((len(adj_arr), TOTAL_NOUN))
        for i, adj in enumerate(adj_arr):
            noun_prob_dist[i, :] = feat[adj]

        noun_prob_dist = noun_prob_dist.var(axis=0)
        mean_noun_prob_dist[j] = noun_prob_dist
    mean_noun_prob_dist = mean_noun_prob_dist.mean(axis=0)

    #normalize
    if delete_arr:
        mean_noun_prob_dist = np.delete(mean_noun_prob_dist, delete_arr)
        mean_noun_prob_dist /= np.sum(mean_noun_prob_dist)
    else:
        mean_noun_prob_dist /= np.sum(mean_noun_prob_dist)
    return mean_noun_prob_dist


def get_adj_var(feat_list, noun_arr, delete_arr=None):
    '''
    helper function to compute average relatedness variance for nouns
    :param feat_list: list of models
    :param noun_arr: list of noun
    :param delete_arr: section of words to remove
    :return: mean variance
    '''

    mean_adj_prob_dist = np.zeros((len(feat_list), TOTAL_ADJ))
    for j, (feat, _) in enumerate(feat_list):
        adj_prob_dist = np.zeros((len(noun_arr), len(feat.keys())))
        for i, adj in enumerate(feat.keys()):
            adj_prob = np.asarray(feat[adj])[noun_arr]
            adj_prob_dist[:, i] = adj_prob
        adj_prob_dist = adj_prob_dist.var(axis=0)
        mean_adj_prob_dist[j] = adj_prob_dist
    mean_adj_prob_dist = mean_adj_prob_dist.mean(axis=0)

    if delete_arr:
        mean_adj_prob_dist = np.delete(mean_adj_prob_dist, delete_arr)
        mean_adj_prob_dist /= np.sum(mean_adj_prob_dist)
    else:
        mean_adj_prob_dist /= np.sum(mean_adj_prob_dist)
    return mean_adj_prob_dist


def gibbs_search(model_list, adj_list, choice_func, num_iter=1000, comp='speaker', prob_var=True, scen_perm=False):
    '''
    Run gibbs search to find high utility scenarios
    :param model_list: list of models to compare in (adj_mat, prag)
    :param adj_list: list of adjectives
    :param choice_func: function to use to select top utility
    :param num_iter: number of iterations, 1000 by default
    :param comp: comparison metric 'speaker' or 'listener'
    :param prob_var: consider sample variance
    :param scen_perm: permute scenes
    :return: heap of top utility scenarios
    '''
    # randomly initialized
    samples = np.random.choice(range(0, TOTAL_NOUN), size=NUM_NOUN, replace=False)
    out_group = [samples[i] for i in range(2, NUM_NOUN)]
    noun_scen = [[samples[0], samples[1]], out_group]

    if prob_var:
        adj_prob_dist = get_adj_var(model_list, samples)
        adj_scen = list(np.random.choice(range(0, TOTAL_ADJ), size=NUM_ADJ, replace=False, p=adj_prob_dist))
    else:
        adj_scen = list(np.random.choice(range(0, TOTAL_ADJ), size=NUM_ADJ, replace=False))

    adj_scen_list = [adj_list[i] for i in adj_scen]
    curr_util = util_total(model_list, adj_scen_list, noun_scen, choice_func, comp=comp, scen_perm=scen_perm)

    max_util_arr = []
    curr_util_arr = []
    accept_prob_arr = []
    scen_heap = []

    heapq.heappush(scen_heap, (curr_util, (noun_scen, adj_scen_list)))

    max_util_arr.append(curr_util)
    curr_util_arr.append(curr_util)
    accept_prob_arr.append(1)

    for i in range(num_iter):
        if i%1000 == 0:
            print(i)
        for n in range(len(samples)):

            x_n = get_noun_sample_arr(model_list=model_list, adj_samples=adj_scen_list, x_sampled=list(samples[:n]) + list(samples[n+1:]),
                                 n_total=TOTAL_NOUN, prob_var=True)
            prev_x = samples[n]
            samples[n] = x_n
            if prev_x != x_n:

                out_group = [samples[i] for i in range(2, NUM_NOUN)]
                new_noun_scen = [[samples[0], samples[1]], out_group]

                # compute new utility
                if len(set(samples)) < NUM_NOUN:
                    print(samples)

                new_util = util_total(model_list, adj_scen_list, new_noun_scen, choice_func, comp=comp, scen_perm=scen_perm)
                curr_util_arr.append(new_util)
                # keep track of max util value
                max_util_val = max(max_util_arr)
                if new_util > max_util_val:
                    max_util_arr.append(new_util)
                else:
                    max_util_arr.append(max_util_val)

                # decide to change sample
                p = 1 if new_util/curr_util > 1 else new_util/curr_util
                accept_prob_arr.append(p)
                switch = np.random.binomial(1, p)
                if switch:
                    curr_util = new_util
                    if len(scen_heap) < HEAP_SIZE:
                        heapq.heappush(scen_heap, (new_util, (new_noun_scen, adj_scen_list)))
                    else:
                        heapq.heappushpop(scen_heap, (new_util, (new_noun_scen, adj_scen_list)))
                else:
                    samples[n] = prev_x

            else:
                accept_prob_arr.append(0)
                max_util_val = max(max_util_arr)
                max_util_arr.append(max_util_val)
                curr_util_arr.append(new_util)

        out_group = [samples[i] for i in range(2, NUM_NOUN)]
        new_noun_scen = [[samples[0], samples[1]], out_group]

        if prob_var:
            adj_prob_dist = get_adj_var(model_list, samples)
            new_adj_scen = list(np.random.choice(range(0, TOTAL_ADJ), size=NUM_ADJ, replace=False, p=adj_prob_dist))
        else:
            new_adj_scen = list(np.random.choice(range(0, TOTAL_ADJ), size=NUM_ADJ, replace=False))

        new_adj_scen_list = [adj_list[i] for i in new_adj_scen]
        for n in range(len(new_adj_scen)):

            a_n = get_adj_sample_arr(model_list=model_list, noun_samples=samples,
                                 x_sampled=list(new_adj_scen[:n]) + list(new_adj_scen[n+1:]), n_total=TOTAL_ADJ, prob_var=prob_var)

            a_n_word = adj_list[a_n]
            prev_a = new_adj_scen_list[n]
            if a_n_word != prev_a:

                new_adj_scen_list[n] = a_n_word
                new_util = util_total(model_list, new_adj_scen_list, new_noun_scen, choice_func, comp=comp, scen_perm=scen_perm)

                curr_util_arr.append(new_util)
                # update max value
                max_util_val = max(max_util_arr)
                if new_util > max_util_val:
                    max_util_arr.append(new_util)
                else:
                    max_util_arr.append(max_util_val)

                # decide to change sample
                p = 1 if new_util/curr_util > 1 else new_util/curr_util
                accept_prob_arr.append(p)
                switch = np.random.binomial(1, p)
                if switch:
                    curr_util = new_util
                    # only want to push onto heap if we are leaving this scenario
                    save_adj_arr = new_adj_scen_list[:]
                    if len(set(save_adj_arr)) < NUM_ADJ:
                        print(save_adj_arr)
                    if len(scen_heap) < HEAP_SIZE:
                        heapq.heappush(scen_heap, (new_util, (new_noun_scen, save_adj_arr)))
                    else:
                        heapq.heappushpop(scen_heap, (new_util, (new_noun_scen, save_adj_arr)))

                    new_adj_scen[n] = a_n
                else:
                    new_adj_scen_list[n] = prev_a
            else:
                accept_prob_arr.append(0)
                max_util_val = max(max_util_arr)
                max_util_arr.append(max_util_val)
                curr_util_arr.append(new_util)

        adj_scen_list = new_adj_scen_list

    logger = [max_util_arr, curr_util_arr, accept_prob_arr]
    return logger, scen_heap


def plot_util(max_util_arr, curr_util_arr, accept_prob_arr, rand_val):
    '''
    helper function to plot utility
    :param max_util_arr: array containing max util at each iteration
    :param curr_util_arr: array countaing sampled util at each iteration
    :param accept_prob_arr: probability of acceptance
    :param rand_val: random threshold
    :return: None
    '''
    # plot progress
    x = np.arange(0, len(max_util_arr))
    plt.figure(1)
    plt.subplot(211)
    line_1 = plt.plot(x, max_util_arr)
    line_2 = plt.plot(x, curr_util_arr)
    # use keyword args
    plt.setp(line_1, label='max_util')
    plt.setp(line_2, label='curr_util')
    plt.axhline(rand_val)
    plt.legend()

    plt.subplot(212)
    plt.plot(x, accept_prob_arr)
    plt.show()


def rand_util(model_list, choice_func, comp='speaker', iter=1000, scen_perm=False):
    '''
    generate randomly sampled threshold value
    :param model_list: list of models to compare in (adj_mat, prag)
    :param choice_func: function to use to select top utility
    :param comp: comparison metric 'speaker' or 'listener'
    :param iter: number of iterations
    :param scen_perm: permute scenes
    :return: return mean util of iter of sampled scenarios
    '''
    rand_util_arr = []
    for i in range(iter):
        samples = np.random.choice(range(0, TOTAL_NOUN), size=NUM_NOUN, replace=False)
        out_group = [samples[i] for i in range(2, NUM_NOUN)]
        noun_scen = [[samples[0], samples[1]], out_group]
        adj_scen = list(np.random.choice(range(0, TOTAL_ADJ), size=NUM_ADJ, replace=False))
        adj_scen_list = [adj_list[i] for i in adj_scen]
        curr_util = util_total(model_list, adj_scen_list, noun_scen, choice_func, comp=comp, scen_perm=scen_perm)
        rand_util_arr.append(curr_util)
    return np.mean(rand_util_arr)


def run_listener_search(adj_list, word_list, num_iter, model_comp_list, comp_list_str, prob_var, scen_perm=False):
    '''
    run listener search using gibbs sampling, save scenarios in ../output
    :param adj_list: list of all possible adjectives
    :param word_list: list of all possible codenames
    :param num_iter: number of iterations
    :param model_comp_list: list of models to compare in (adj_mat, prag)
    :param comp_list_str: names of models
    :param prob_var: incorporate answer variance
    :param scen_perm: permute scenes
    :return: None
    '''
    # listener Gibbs search
    logger, max_scen_heap = gibbs_search(model_comp_list, adj_list, listener_choice_given_M_D, num_iter=num_iter,
                                         comp='listener', prob_var=prob_var, scen_perm=scen_perm)

    max_util_arr, curr_util_arr, accept_prob_arr = logger
    f = open('listener_log.txt', 'w')
    print("mean acceptance prob", np.mean(accept_prob_arr))

    save_scenario_arr = []
    for val, obj in heapq.nlargest(HEAP_SIZE, max_scen_heap):
        f.write(" \n")
        f.write(' '.join(['utility:', str(val), '\n']))
        out_group = [obj[0][1][i] for i in range(len(obj[0][1]))]
        in_group = obj[0][0]
        scen = in_group + out_group
        f.write(' '.join(['Codenames: '] + [word_list[scen[i]] for i in range(len(scen))] + ['\n']))
        adjs = obj[1]
        f.write('Clues ' + ' '.join(adjs) + '\n')
        f.write(' '.join(['Clue given: ', adjs[0], '\n']))
        save_scenario_arr.append([in_group, out_group, adjs])
        pair_prob = average_across_models(model_comp_list, obj[1], obj[0], listener_choice_given_M_D,
                                          comp='listener', idx=0)
        noun_pairs = list(itertools.combinations(scen, 2))
        for i, noun_pair in enumerate(noun_pairs):
            f.write(' '.join(
                [word_list[noun_pairs[i][0]] + ' ' + word_list[noun_pairs[i][1]], str(round(pair_prob[i], 3)),
                 '\n']))

        model_choices = abs_max_across_models(model_comp_list, obj[1], obj[0], comp='listener', idx=0)
        for i in range(len(model_comp_list)):
            f.write(' '.join([comp_list_str[i] + ' answer: ', word_list[noun_pairs[model_choices[i]][0]],
                              word_list[noun_pairs[model_choices[i]][1]], '\n']))
    f.close()
    with open('../output/gibbs_bigram_listener' + str(num_iter) + '.pickle', 'wb') as f:
        pickle.dump({'scenarios': save_scenario_arr}, f, protocol=2)


def run_speaker_search(adj_list, word_list, num_iter, model_comp_list, comp_list_str, prob_var, scen_perm=False):
    '''
    run speaker search using gibbs sampling, save scenarios in ../output
    :param adj_list: list of all possible adjectives
    :param word_list: list of all possible codenames
    :param num_iter: number of iterations
    :param model_comp_list: list of models to compare in (adj_mat, prag)
    :param comp_list_str: names of models
    :param prob_var: incorporate answer variance
    :param scen_perm: permute scenes
    :return: None
    '''
    logger, max_scen_heap = gibbs_search(model_comp_list, adj_list, speaker_choice_given_M_D, num_iter=num_iter,
                                         prob_var=prob_var, scen_perm=scen_perm)
    max_util_arr, curr_util_arr, accept_prob_arr = logger

    f =  open('speaker_log.txt', 'w')

    print("mean acceptance prob", np.mean(accept_prob_arr))
    save_scenario_arr = []
    for val, obj in heapq.nlargest(HEAP_SIZE, max_scen_heap):
        f.write(' \n')
        f.write(' '.join(['utility: ', str(val), '\n']))
        out_group = [obj[0][1][i] for i in range(len(obj[0][1]))]
        in_group = obj[0][0]
        scen = in_group + out_group
        f.write(' '.join(['Codenames: '] + [word_list[scen[i]] for i in range(len(scen))] + ['\n']))
        adjs = obj[1]
        save_scenario_arr.append([in_group, out_group, adjs])
        adj_prob = average_across_models(model_comp_list, obj[1], obj[0], speaker_choice_given_M_D)
        f.write("clue , p(clue) \n")
        for i, adj in enumerate(adjs):
            f.write(' '.join([adjs[i], str(round(adj_prob[i], 3)), '\n']))

        model_choices = abs_max_across_models(model_comp_list, obj[1], obj[0])
        for i in range(len(model_comp_list)):
            f.write(' '.join([comp_list_str[i] + ' answer: ', adjs[model_choices[i]], '\n']))
    f.close()
    with open('../output/gibbs_pragmatic_speaker'+ str(num_iter) + '.pickle', 'wb') as f:
        pickle.dump({'scenarios': save_scenario_arr}, f, protocol=2)


def run_comb_search(adj_list, word_list, num_iter, model_comp_list, comp_list_str, prob_var, scen_perm=False):
    '''
    run speaker and listener together, save scenarios in ../output
    :param adj_list: list of all possible adjectives
    :param word_list: list of all possible codenames
    :param num_iter: number of iterations
    :param model_comp_list: list of models to compare in (adj_mat, prag)
    :param comp_list_str: names of models
    :param prob_var: incorporate answer variance
    :param scen_perm: permute scenes
    :return: None
    '''

    logger, max_scen_heap = gibbs_search(model_comp_list, adj_list, (speaker_choice_given_M_D, listener_choice_given_M_D),
                                         num_iter=num_iter, prob_var=prob_var, scen_perm=scen_perm, comp='all')

    max_util_arr, curr_util_arr, accept_prob_arr = logger

    f = open('combined_log.txt', 'w')
    model_list_listener = [(model, prag[0] + 'l') for (model, prag) in model_comp_list]
    model_list_speaker = [(model, prag[0] + 's') for (model, prag) in model_comp_list]
    print("mean acceptance prob", np.mean(accept_prob_arr))
    save_scenario_arr = []
    for k, (val, obj) in enumerate(heapq.nlargest(HEAP_SIZE, max_scen_heap)):
        f.write("scenario: %d"%k)
        f.write(' \n')
        f.write(' '.join(['utility: ', str(val), '\n']))
        out_group = [obj[0][1][i] for i in range(len(obj[0][1]))]
        in_group = obj[0][0]
        scen = in_group + out_group
        noun_pairs = list(itertools.combinations(scen, 2))
        f.write(' '.join(['Codenames: '] + [word_list[scen[i]] for i in range(len(scen))] + ['\n']))
        adjs = obj[1]
        save_scenario_arr.append([in_group, out_group, adjs])
        for j in range(NUM_NPAIR):
            f.write(' '.join(
                [word_list[noun_pairs[j][0]] + ' ' + word_list[noun_pairs[j][1]], '\n']))

            adj_prob = average_across_models(model_list_speaker, obj[1], obj[0], speaker_choice_given_M_D, comp='speaker', idx=j)
            # listener scenario
            f.write("clue , p(clue) \n")
            for i, adj in enumerate(adjs):
                f.write(' '.join([adjs[i], str(round(adj_prob[i], 3)), '\n']))

            model_choices = abs_max_across_models(model_list_speaker, obj[1], obj[0], idx=j)
            for i in range(len(model_list_speaker)):
                f.write(' '.join([comp_list_str[i] + ' answer: ', adjs[model_choices[i]], '\n']))
            f.write(' \n')

            model_dist = dist_across_models(model_list_speaker, adjs, [in_group, out_group], comp='speaker', idx=j)
            for i in range(len(model_list_speaker)):
                f.write(' '.join([model_comp_list_str[i] + ' answer: ', str(np.round_(model_dist[i], 3)), '\n']))
            f.write(' \n')

        for j in range(NUM_ADJ):
            f.write(' '.join(['Clue given: ', adjs[j], '\n']))
            save_scenario_arr.append([in_group, out_group, adjs])
            pair_prob = average_across_models(model_list_listener, obj[1], obj[0], listener_choice_given_M_D,
                                              comp='listener', idx=j)

            for i, noun_pair in enumerate(noun_pairs):
                f.write(' '.join(
                    [word_list[noun_pairs[i][0]] + ' ' + word_list[noun_pairs[i][1]], str(round(pair_prob[i], 3)),
                     '\n']))

            model_choices = abs_max_across_models(model_list_listener, obj[1], obj[0], comp='listener', idx=j)
            for i in range(len(model_list_listener)):
                f.write(' '.join([comp_list_str[i] + ' answer: ', word_list[noun_pairs[model_choices[i]][0]],
                                  word_list[noun_pairs[model_choices[i]][1]], '\n']))
            f.write(' \n')

            model_dist = dist_across_models(model_list_listener, adjs, [in_group, out_group], comp='listener', idx=j)
            for i in range(len(model_list_listener)):
                f.write(' '.join([model_comp_list_str[i] + ' answer: ', str(np.round_(model_dist[i], 3)), '\n']))
                f.write(' \n')

    f.close()
    with open('../output/gibbs_pragmatic_comb' + str(num_iter) + '.pickle', 'wb') as f:
        pickle.dump({'scenarios': save_scenario_arr}, f, protocol=2)


def eval_existing_scenarios(word_list, glove_feats):
    '''
    evaluate utility of 8 adjective 5 noun set of scenarios
    :param word_list: list of words
    :param glove_feats: glove features
    :return: None
    '''
    util_arr_lit = []
    word_scen_arr = []
    abs_result = []
    for i in range(80):
        [n_0, n_1], [n_2, n_3, n_4] = scenarios[i]
        scen_str = ' ( ' + word_list[n_0] + ' ' +  word_list[n_1] + ' ) ' \
                   + word_list[n_2] + ' ' + word_list[n_3] + ' '+ word_list[n_4]
        word_scen_arr.append(scen_str)
        util_arr_lit.append(np.max(util_total([bigram_feats, conceptnet_feats, glove_feats], scenario_adjs[i], scenarios[i],
                                       listener_choice_given_M_D, comp='ll')))
        abs_arr = [abs_max_across_models([bigram_feats, conceptnet_feats, glove_feats], scenario_adjs[i], scenarios[i],
                                         comp='ll', idx=k) for k in range(NUM_ADJ)]
        abs_result.append(abs_arr)

    last5_util = np.argsort(util_arr_lit)[:5]
    top5_util = np.argsort(util_arr_lit)[-5:]

    print("mean listener util", np.mean(util_arr_lit))

    print("Top 5: ")
    for ind in top5_util:
        print(word_scen_arr[ind], scenario_adjs[ind])
        print(util_arr_lit[ind], abs_result[ind])

    print("Last 5: ")
    for ind in last5_util:
        print(word_scen_arr[ind], scenario_adjs[ind])
        print(util_arr_lit[ind], abs_result[ind])

    speaker_file = ['../data/speaker_glove.csv', '../data/speaker_bigram.csv']
    speaker_matrix, scenario_count = data_utils.parse_speaker(speaker_file)
    speaker_entropy = entropy.get_data_entropy(speaker_matrix, num_dim=2)

    util_arr_lit = []
    util_arr_prag = []
    word_scen_arr = []
    abs_result = []
    for i in range(80):
        [n_0, n_1], [n_2, n_3, n_4] = scenarios[i]
        scen_str = ' ( ' + word_list[n_0] + ' ' +  word_list[n_1] + ' ) ' \
                   + word_list[n_2] + ' ' + word_list[n_3] + ' '+ word_list[n_4]
        word_scen_arr.append(scen_str)
        util_arr_lit.append(util_total([bigram_feats, conceptnet_feats, glove_feats], scenario_adjs[i], scenarios[i],
                                       speaker_choice_given_M_D, comp='ls'))
        util_arr_prag.append(util_total([bigram_feats, conceptnet_feats, glove_feats], scenario_adjs[i], scenarios[i],
                                        speaker_choice_given_M_D, comp='ps'))
        abs_result.append(abs_max_across_models([bigram_feats, conceptnet_feats, glove_feats], scenario_adjs[i], scenarios[i]))
    last5_util = np.argsort(util_arr_lit)[:5]
    top5_util = np.argsort(util_arr_lit)[-5:]

    print("mean sepaker util", np.mean(util_arr_lit))

    print("Top 5: ")
    for ind in top5_util:
        print(word_scen_arr[ind], scenario_adjs[ind])
        print(util_arr_lit[ind], abs_result[ind])

    print("Last 5: ")
    for ind in last5_util:
        print(word_scen_arr[ind], scenario_adjs[ind])
        print(util_arr_lit[ind], abs_result[ind])


def print_results():
    feature_str = 'google_bigram_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        bigram_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'conceptnet_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        conceptnet_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'google_w2v_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        w2v_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'lda_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        lda_feats = pickle.load(fp, encoding='latin-1')

    with open('../output/gibbs_semantic_geo_comb100000.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    model_comp_list = [(bigram_feats, 'll'), (conceptnet_feats, 'll'), (w2v_feats, 'll'), (lda_feats, 'll')]
    model_comp_list_str = ['bigram_feats', 'conceptnet_feats', 'w2v_feats', 'lda_feats']

    f = open('combined_log_prob.txt', 'w')
    for in_group, out_group, adjs in scenarios:
        f.write(' \n')
        scen = in_group + out_group
        noun_pairs = list(itertools.combinations(scen, 2))
        f.write(' '.join(['Codenames: '] + [word_list[scen[i]] for i in range(len(scen))] + ['\n']))

        for j in range(NUM_NPAIR):
            f.write(' '.join(
                [word_list[noun_pairs[j][0]] + ' ' + word_list[noun_pairs[j][1]], '\n']))
            adj_prob = average_across_models(model_comp_list, adjs, [in_group, out_group], speaker_choice_given_M_D, comp='speaker', idx=j)

            # listener scenario
            f.write("clue , p(clue) \n")
            for i, adj in enumerate(adjs):
                f.write(' '.join([adjs[i], str(round(adj_prob[i], 3)), '\n']))
            model_choices = dist_across_models(model_comp_list, adjs, [in_group, out_group], idx=j)

            for i in range(len(model_comp_list)):
                f.write(' '.join([model_comp_list_str[i] + ' answer: ', str(np.round_(model_choices[i], 3)), '\n']))
            f.write(' \n')

        for j in range(NUM_ADJ):
            f.write(' '.join(['Clue given: ', adjs[j], '\n']))
            pair_prob = average_across_models(model_comp_list, adjs, [in_group, out_group], listener_choice_given_M_D,
                                              comp='listener', idx=j)

            for i, noun_pair in enumerate(noun_pairs):
                f.write(' '.join(
                    [word_list[noun_pairs[i][0]] + ' ' + word_list[noun_pairs[i][1]], str(round(pair_prob[i], 3)),
                     '\n']))

            model_choices = dist_across_models(model_comp_list, adjs, [in_group, out_group], comp='listener', idx=j)
            for i in range(len(model_comp_list)):
                f.write(' '.join([model_comp_list_str[i] + ' answer: ', str(np.round_(model_choices[i], 3)), '\n']))
            f.write(' \n')

    f.close()


def main_search(adj_list, word_list, mode='semantic'):
    feature_str = 'google_bigram_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        bigram_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'conceptnet_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        conceptnet_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'google_w2v_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        w2v_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'lda_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        lda_feats = pickle.load(fp, encoding='latin-1')

    if mode == 'semantic':
        model_comp_list = [(bigram_feats, 'll'), (conceptnet_feats, 'll'), (w2v_feats, 'll'), (lda_feats, 'll')]
        model_comp_list_str = ['bigram_feats', 'conceptnet_feats', 'w2v_feats', 'lda_feats']

    elif mode == 'pragmatic':
        model_comp_list = [(bigram_feats, 'll'), (bigram_feats, 'pl')]
        model_comp_list_str = ['bigram_literal', 'bigramt_pragmatic']
    else:
        print("invalid mode input")
        return

    run_comb_search(adj_list, word_list, num_iter=5000, model_comp_list=model_comp_list,
                    comp_list_str=model_comp_list_str, prob_var=False, scen_perm=True)

if __name__ == "__main__" :
    data_set_str = 'combined'
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'

    if data_set_str == 'glove':
        adjective_file = 'adj_glove.pickle'
    else:
        adjective_file = 'adj_choosen.pickle'

    word_list, adj_list, adj_code_prob_dict, scenarios, scenario_adjs = \
            data_utils.load_pickle_files(file_path, codenames_file, adjective_file)

    main_search(adj_list, word_list, mode='semantic')