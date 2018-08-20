import _pickle as pickle
import numpy as np
import csv
import itertools
import seaborn as sns
import pandas as pd
import json
import random
import data_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import warnings
from distance_utils import wasserstein, jsd, topk_dist, topk_avg_acc, topk_acc, entropy
import experiment_utils
import scipy.stats
import visualization_utils as viz_utils

sns.set_style(rc={'axes.axisbelow': False, 'legend.frameon': False})
warnings.filterwarnings('error')
random.seed(42)

NUM_ADJ = 3
NUM_NPAIRS = 3
PLOT_HISTOGRAMS = False
PLOT_PROB = True
SAVE_SIMULATIONS = True
scalar = MinMaxScaler()

def extract_scenario_table(adj_scen, scen, adj_feat, object_agg_func):
    '''
    Construct adjective scenario table from data
    :param adj_scen: scenario adjectives
    :param scen: scenario nouns
    :param adj_feat: adjective features
    :param object_agg_func: aggregation function to calculate joint probability
    :return: matrix of adjective vs noun pair probabilities
    '''
    adj_list_init = adj_scen
    out_group = [scen[1][i] for i in range(len(scen[1]))]
    in_group = [scen[0][0], scen[0][1]]
    sample_list = in_group + out_group
    n_0 = sample_list[0]
    n_1 = sample_list[1]

    noun_pairs = list(itertools.combinations(sample_list, 2))
    raw = np.zeros((len(adj_scen), len(noun_pairs)))

    for i in range(len(adj_scen)):
        adj = adj_list_init[i]
        for j in range(NUM_NPAIRS):
            (ind_0, ind_1) = noun_pairs[j]
            raw[i,j] = object_agg_func([adj_feat[adj][ind_0], adj_feat[adj][ind_1]])
    return raw, noun_pairs.index((n_0, n_1))

def raw_scenario_table(adj_scen, noun_scen, adj_feat):
    '''
    extract raw scenario table by adjective
    :param adj_scen: scenario adjectives
    :param noun_scen: scenario nouns
    :param adj_feat: adjective features
    :return: raw matrix probabilities
    '''
    raw = np.zeros((len(adj_scen), len(noun_scen)))
    for i, adj in enumerate(adj_scen):
        for j, n_ind in enumerate(noun_scen):
            raw[i, j] = adj_feat[adj][n_ind]
    return raw

def rsa_listener(raw, alpha):
    '''
    Compute pragmatic listener from raw probabilities
    :param raw: input matrix of adjectives vs noun pairs
    :param alpha: pragmatic coefficient
    :return: dictionary of literal listener, pragmatic speaker and pragmatic listener
    '''
    # normalize
    literal_listener = raw / raw.sum(axis=1, keepdims=True)

    # take agent's rationality into account and normalize
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'divide by zero encountered in log')
        speaker_utility = np.log(literal_listener)
    literal_speaker = np.exp(alpha * speaker_utility)
    literal_speaker /= literal_speaker.sum(axis=0, keepdims=True)

    # normalize
    pragmatic_listener = literal_speaker / literal_speaker.sum(axis=1, keepdims=True)
    return {'ll':literal_listener, 'ls':literal_speaker, 'pl':pragmatic_listener}


def rsa_speaker(raw, alpha):
    '''
    Compute pragmatic speaker from raw probabilities
    :param raw: input matrix of adjectives vs noun pairs
    :param alpha: pragmatic coefficient
    :return: dictionary of literal speaker, pragmatic listener and pragmatic speaker
    '''
    # normalize
    literal_speaker = raw / raw.sum(axis=0, keepdims=True)

    # take agent's rationality into account and normalize
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'divide by zero encountered in log')
        listener_utility = np.log(literal_speaker)
    literal_listener = np.exp(alpha * listener_utility)
    try:
        literal_listener /= literal_listener.sum(axis=1, keepdims=True)
    except Warning:
        print(literal_listener)

    # normalize
    pragmatic_speaker = literal_listener / literal_listener.sum(axis=0, keepdims=True)
    return {'ls':literal_speaker, 'll':literal_listener, 'ps':pragmatic_speaker}


def extract_samples(scen):
    '''
    extract word string information from scenario
    :param scen: input scenario of nouns
    :param word_list: list of words
    :return: noun indices, noun pairs, ingroup, out group
    '''
    out_group = [scen[1][i] for i in range(len(scen[1]))]
    in_group = scen[0]
    n_0, n_1 = in_group
    sample_list = in_group + out_group

    samples = list(itertools.combinations(sample_list, 2))
    samp_ind_ingroup = samples.index((n_0, n_1))

    return samples, samp_ind_ingroup


def listener_sub(a_i, rsa_matrices, mturk_scen_listener, word_list, adj_scen, samples, scen_num, ent_limit, verbose=False):
    '''
    helper method for computing model accuracy and distance to original distribution
    :param a_i: adjective index
    :param rsa_matrices: rsa probability matricies
    :param mturk_scen_listener: mturk distribution
    :param word_list: list of all nouns
    :param adj_scen: adjective set
    :param samples: noun paires
    :param scen_num: scenario index
    :param ent_limit: entropy upper bound
    :param verbose: whether to suppress output
    :return: wd_lit, wd_prag, wd_uniform, kl_lit, kl_prag, kl_uniform, hit_lit, hit_prag, hist['mturk'], \
           hist['literal'], hist['pragmatic'], bounded_entropy
    '''
    bounded_entropy = True
    if ent_limit is not None and entropy(mturk_scen_listener[a_i]) > ent_limit:
        if not SAVE_SIMULATIONS:
            return [np.nan for _ in range(8)]

    hist = OrderedDict()
    hist['mturk'] = mturk_scen_listener[a_i]
    hist['literal'] = rsa_matrices['ll'][a_i]
    hist['pragmatic'] = rsa_matrices['pl'][a_i]

    wd_lit = wasserstein(mturk_scen_listener[a_i], rsa_matrices['ll'][a_i])
    wd_prag = wasserstein(mturk_scen_listener[a_i], rsa_matrices['pl'][a_i])
    wd_uniform = wasserstein(mturk_scen_listener[a_i], np.ones(NUM_NPAIRS) / NUM_NPAIRS)
    kl_lit = jsd(mturk_scen_listener[a_i], rsa_matrices['ll'][a_i])
    kl_prag = jsd(mturk_scen_listener[a_i], rsa_matrices['pl'][a_i])
    kl_uniform = jsd(mturk_scen_listener[a_i], np.ones(NUM_NPAIRS) / NUM_NPAIRS)

    if verbose:
        print(adj_scen[a_i] + '\n')

    # find the top human-rated if there's one and see how far away the model was
    top_mturk = np.argmax(mturk_scen_listener[a_i])

    if (entropy(mturk_scen_listener[a_i]) <= ent_limit) and sum(mturk_scen_listener[a_i] == mturk_scen_listener[a_i, top_mturk]) == 1:
        hit_lit = topk_dist(top_mturk, rsa_matrices['ll'][a_i])
        hit_prag = topk_dist(top_mturk, rsa_matrices['pl'][a_i])
    else:
        bounded_entropy = False
        hit_lit = np.nan
        hit_prag = np.nan

    if PLOT_HISTOGRAMS:
        label_cat = np.array(["{}_{}".format(word_list[s[0]], word_list[s[1]]) for s in samples])
        adj_scen = [adj for adj in adj_scen]
        plot_bargraph(hist, label_cat, 'listener', str(scen_num) + '_' + adj_scen[a_i] + '_all_' + '_'.join(adj_scen))

    if verbose:
        for i in range(NUM_NPAIRS):
            pair1, pair2 = samples[i]
            print(word_list[pair1] + '_' + word_list[pair2])
            print(round(rsa_matrices['ll'][a_i, i], 3))
            print(round(rsa_matrices['pl'][a_i, i], 3))
            print(round(mturk_scen_listener[a_i, i], 3))

    return wd_lit, wd_prag, wd_uniform, kl_lit, kl_prag, kl_uniform, hit_lit, hit_prag, hist['mturk'].tolist(), \
           hist['literal'].tolist(), hist['pragmatic'].tolist(), bounded_entropy


def comparison_rsa_listener(rsa_matrices, scen, adj_scen, mturk_scen_listener, word_list, scen_num, ent_limit, verbose=0,
                            results_dict=None, single_adj=False):
    """ User is given a word list, an adjective list and a single selected adjective.
        They must select the noun combination that makes sense. """

    samples, samp_ind_ingroup = extract_samples(scen)

    wd_lit = []
    wd_prag = []
    wd_uniform = []
    kl_lit = []
    kl_prag = []
    kl_uniform = []
    hit_lit = []
    hit_prag = []
    if results_dict:
        (mturk_dict, literal_dict, prag_dict, entropy_dict) = results_dict
    for a_i in range(len(adj_scen)):
        sub = listener_sub(a_i, rsa_matrices, mturk_scen_listener, word_list, adj_scen, samples, scen_num, ent_limit, verbose)
        wd_lit.append(sub[0])
        wd_prag.append(sub[1])
        wd_uniform.append(sub[2])
        kl_lit.append(sub[3])
        kl_prag.append(sub[4])
        kl_uniform.append(sub[5])
        hit_lit.append(sub[6])
        hit_prag.append(sub[7])
        if results_dict:
            mturk_dict[adj_scen[a_i]] = {}
            literal_dict[adj_scen[a_i]] = {}
            prag_dict[adj_scen[a_i]] = {}
            if SAVE_SIMULATIONS:
                for w_i in range(NUM_NPAIRS):
                    mturk_dict[adj_scen[a_i]][str(samples[w_i])] = sub[8][w_i]
                    literal_dict[adj_scen[a_i]][str(samples[w_i])] = sub[9][w_i]
                    prag_dict[adj_scen[a_i]][str(samples[w_i])] = sub[10][w_i]
                entropy_dict[adj_scen[a_i]] = sub[11]
        if verbose:
            print("\n\n\n")
        if single_adj:
            break
    return {'lit': wd_lit, 'prag': wd_prag, 'uniform': wd_uniform}, \
           {'lit': kl_lit, 'prag': kl_prag, 'uniform': kl_uniform}, \
           {'lit': hit_lit, 'prag': hit_prag}


def comparison_rsa_speaker(rsa_matrices, scen, adj_scen, mturk_scen_speaker, word_list, scen_num, ent_limit, verbose=0, results_dict=None):
    """ User is given a word list and an adjective list.
        They must select an adjective that makes sense. """

    samples, samp_ind_ingroup = extract_samples(scen)

    if ent_limit is not None and entropy(mturk_scen_speaker) > ent_limit:
        if SAVE_SIMULATIONS == False:
            return {'lit': np.nan, 'prag': np.nan, 'uniform': np.nan}, \
                   {'lit': np.nan, 'prag': np.nan, 'uniform': np.nan}, \
                   {'lit': np.nan, 'prag': np.nan}


    hist = OrderedDict()
    hist['mturk'] = mturk_scen_speaker
    hist['literal'] = rsa_matrices['ls'][:, samp_ind_ingroup]
    hist['pragmatic'] = rsa_matrices['ps'][:, samp_ind_ingroup]
    hist['literal_listener'] = rsa_matrices['ll'][:, samp_ind_ingroup]

    if results_dict:
        (mturk_dict, literal_dict, prag_dict, entropy_dict) = results_dict

    # find the top human-rated if there's one and see how far away the model was
    top_mturk = np.argmax(mturk_scen_speaker)
    if (entropy(mturk_scen_speaker) <= ent_limit) and sum(mturk_scen_speaker == mturk_scen_speaker[top_mturk]) == 1:
        hit_lit = topk_dist(top_mturk, hist['literal'])
        hit_prag = topk_dist(top_mturk, hist['pragmatic'])
    else:
        hit_lit = np.nan
        hit_prag = np.nan

    if results_dict:
        for i in range(len(hist['mturk'])):
            mturk_dict[adj_scen[i]] = hist['mturk'][i]
            literal_dict[adj_scen[i]] = hist['literal'][i]
            prag_dict[adj_scen[i]] = hist['pragmatic'][i]

            if ent_limit is not None:
                if (entropy(mturk_scen_speaker) <= ent_limit) and sum(mturk_scen_speaker == mturk_scen_speaker[top_mturk]) == 1:
                    entropy_dict[adj_scen[i]] = True
                else:
                    entropy_dict[adj_scen[i]] = False


    wd_lit = wasserstein(mturk_scen_speaker, hist['literal'])
    wd_prag = wasserstein(mturk_scen_speaker, hist['pragmatic'])
    wd_uniform = wasserstein(mturk_scen_speaker, np.ones(NUM_ADJ) / NUM_ADJ)
    kl_lit = jsd(mturk_scen_speaker, hist['literal'])
    kl_prag = jsd(mturk_scen_speaker, hist['pragmatic'])
    kl_uniform = jsd(mturk_scen_speaker, np.ones(NUM_ADJ) / NUM_ADJ)


    if PLOT_HISTOGRAMS:
        plot_bargraph(hist, adj_scen, 'speaker', str(scen_num) + '_ingroup_' + ingroup_str + '_outgroup_' + outgroup_str)


    for i in range(NUM_ADJ):
        if verbose:
            print(adj_scen[i])
            print(round(rsa_matrices['ls'][i, samp_ind_ingroup], 3))
            print(round(rsa_matrices['ps'][i, samp_ind_ingroup], 3))
            print(round(mturk_scen_speaker[i], 3))
    assert 0.99 < sum(rsa_matrices['ls'][:, samp_ind_ingroup]) < 1.01
    assert 0.99 < sum(rsa_matrices['ps'][:, samp_ind_ingroup]) < 1.01
    if verbose:
        print("\n\n\n")
    return {'lit': wd_lit, 'prag': wd_prag, 'uniform': wd_uniform}, \
           {'lit': kl_lit, 'prag': kl_prag, 'uniform': kl_uniform}, \
           {'lit': hit_lit, 'prag': hit_prag}


def analyze_speaker(mturk_speaker_scenarios, word_list, scenarios_nouns, scenario_adjs, adj_feats, object_agg_func, ent_limit, alpha):
    '''
    Load speaker data and compare with rsa models
    :param adj_list: list of adjectives
    :param word_list: list of nouns
    :param scenarios_nouns: list of scenarios nouns
    :param scenario_adjs: list of scenario adjectives
    :param adj_feats: noun adjectives similarities
    :param object_agg_func: aggregation function for combining word pair probabilities
    :param ent_limit: entropy upper bound
    :param alpha: pragmatic coefficient
    :return: [top 1 lit acc, top 1 prag acc, top 3 lit acc, top 3 prag acc, lit JSD, prag JSD, lit WD, prag WD]
    '''
    results_dict = {}
    results_dict['mturk'] = {}
    results_dict['ls'] = {}
    results_dict['ps'] = {}
    results_dict['entropy'] = {}

    print("\n-- Speaker --")
    r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7 = analyze_rsa(word_list, scenarios_nouns, scenario_adjs,
               adj_feats, mturk_speaker_scenarios, topk_acc, rsa_speaker,
               comparison_rsa_speaker, object_agg_func, ent_limit, alpha,
                (results_dict['mturk'], results_dict['ls'], results_dict['ps'], results_dict['entropy']))

    print("\n-- Listener Speaker --")
    l_0, l_1, l_2, l_3 = analyze_rsa_literal_speaker(word_list, scenarios_nouns, scenario_adjs,
               adj_feats, mturk_speaker_scenarios, topk_acc, object_agg_func, ent_limit, alpha)

    if SAVE_SIMULATIONS:
        with open('../output/prob_dictionary/speaker_' + 'bigram' + '.json', 'w') as fp:
            json.dump(results_dict, fp)

    return [r_0[0], r_0[1], r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1],
            r_4, r_5, r_6, r_7], [l_0[0], l_0[1],  l_1[0], l_1[1], l_2, l_3]


def get_mturk_listener_data(listener_file, word_list, scenarios_nouns, scenario_adjs, single_adj):
    if isinstance(listener_file, str):
        listener_dset = data_utils.parse_listener(scenarios_nouns[:40], [listener_file], scenario_adjs, word_list,
                                                  with_pair_labels=False, single_adj=single_adj)
    else:
        listener_a = data_utils.parse_listener(scenarios_nouns[:40], [listener_file[0]], scenario_adjs[:40], word_list, with_pair_labels=False)
        listener_b = data_utils.parse_listener(scenarios_nouns[40:], [listener_file[1]], scenario_adjs[40:], word_list,
                                               with_pair_labels=False, single_adj=single_adj)
        listener_dset = np.concatenate((listener_a, listener_b), axis=0)

    def extract_norm(mscen):
        mscen = np.array(mscen)
        return mscen / mscen.sum(axis=1, keepdims=True)
    scenarios_mturks = [extract_norm(listener_dset[i]) for i in range(len(listener_dset))]
    return scenarios_mturks


def analyze_listener(mturk_listener_scenarios, word_list, scenarios_nouns, scenario_adjs, adj_feats, object_agg_func, ent_limit,
                     alpha, single_adj):
    '''
    parse and load listener data and compare with model outputs
    :param adj_list: list of adjectives
    :param word_list: list of nouns
    :param scenarios_nouns: list of scenarios nouns
    :param scenario_adjs: list of scenario adjectives
    :param adj_feats: noun adjectives similarities
    :param object_agg_func: aggregation function for combining word pair probabilities
    :param ent_limit: entropy upper bound
    :param alpha: pragmatic coefficient
    :return: [top 1 lit acc, top 1 prag acc, top 3 lit acc, top 3 prag acc, lit JSD, prag JSD, lit WD, prag WD]
    '''
    results_dict = {}
    results_dict['mturk'] = {}
    results_dict['ll'] = {}
    results_dict['pl'] = {}
    results_dict['entropy'] = {}
    print("\n-- Listener --")
    r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7 = analyze_rsa(word_list, scenarios_nouns, scenario_adjs,
               adj_feats, mturk_listener_scenarios, topk_avg_acc, rsa_listener,
               comparison_rsa_listener, object_agg_func, ent_limit, alpha,
                (results_dict['mturk'], results_dict['ll'], results_dict['pl'], results_dict['entropy']), single_adj=single_adj)

    print("\n-- Speaker - Listener --")
    l_0, l_1, l_2, l_3 = analyze_rsa_literal_listener(word_list, scenarios_nouns, scenario_adjs,
               adj_feats, mturk_listener_scenarios, topk_avg_acc, object_agg_func, ent_limit, alpha, single_adj=single_adj)

    if SAVE_SIMULATIONS:
        with open('../output/prob_dictionary/listener_' + 'bigram' + '.json', 'w') as fp:
            json.dump(results_dict, fp)

    return [r_0[0], r_0[1], r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1],
            r_4, r_5, r_6, r_7], [l_0[0], l_0[1],  l_1[0], l_1[1], l_2, l_3]


def analyze_rsa_literal_speaker(word_list, scenarios_nouns, scenario_adjs, adj_feats,
                mturk_speaker_scenarios, acc_func, object_agg_func, ent_limit, alpha):
    '''
    :param adj_list: list of adjectives
    :param word_list: list of nouns
    :param scenarios_nouns: list of scenarios nouns
    :param scenario_adjs: list of scenario adjectives
    :param adj_feats: noun adjectives similarities
    :param object_agg_func: aggregation function for combining word pair probabilities
    :param ent_limit: entropy upper bound
    :param alpha: pragmatic coefficient
    :return: [top 1 lit acc, top 1 prag acc, top 3 lit acc, top 3 prag acc, lit JSD, prag JSD, lit WD, prag WD]
    '''
    wd_lit = []
    kl_lit = []
    top_dist_lit = []

    for k in range(len(scenario_adjs)):
        raw, _ = extract_scenario_table(scenario_adjs[k], scenarios_nouns[k], adj_feats, object_agg_func)
        rsa_matrices_listener = rsa_listener(raw, alpha)
        rsa_matrices_speaker = rsa_speaker(raw, alpha)

        rsa_matrices_speaker['ls'] = []
        rsa_matrices_speaker['ls'] = rsa_matrices_listener['ls']

        wd_scen, kl_scen, top_dist_scen = \
            comparison_rsa_speaker(rsa_matrices_speaker, scenarios_nouns[k], scenario_adjs[k], mturk_speaker_scenarios[k], word_list, k, ent_limit)

        wd_lit.append(wd_scen['lit'])
        kl_lit.append(kl_scen['lit'])
        top_dist_lit.append(top_dist_scen['lit'])

    print("(S) Literal top 1 hit: {}".format(acc_func(top_dist_lit, 1)))
    print("(S) Literal top 3 hit: {}".format(acc_func(top_dist_lit, 3)))
    print("")
    print("(S) Literal JSD: {}".format(round(np.nanmean(kl_lit), 2)))
    print("")
    print("(S) Literal WD: {}".format(round(np.nanmean(wd_lit), 2)))

    return acc_func(top_dist_lit, 1), acc_func(top_dist_lit, 3), round(np.nanmean(kl_lit), 2), round(np.nanmean(wd_lit), 2)


def analyze_rsa_literal_listener(word_list, scenarios_nouns, scenario_adjs, adj_feats,
                mturk_listener_scenarios, acc_func, object_agg_func, ent_limit, alpha, single_adj=False):
    '''
    :param adj_list: list of adjectives
    :param word_list: list of nouns
    :param scenarios_nouns: list of scenarios nouns
    :param scenario_adjs: list of scenario adjectives
    :param adj_feats: noun adjectives similarities
    :param object_agg_func: aggregation function for combining word pair probabilities
    :param ent_limit: entropy upper bound
    :param alpha: pragmatic coefficient
    :param rsa_dset_func: function to compare data (rsa_listener or rsa_speaker)
    :param rsa_comparison_func: function to comparison_rsa_listener or comparison_rsa_speaker
    :param results_dict: dictionary to store results
    :return: [top 1 lit acc, top 1 prag acc, top 3 lit acc, top 3 prag acc, lit JSD, prag JSD, lit WD, prag WD]
    '''
    wd_lit = []
    kl_lit = []
    top_dist_lit = []

    for k in range(len(scenario_adjs)):
        raw, _ = extract_scenario_table(scenario_adjs[k], scenarios_nouns[k], adj_feats, object_agg_func)
        rsa_matrices_speaker = rsa_speaker(raw, alpha)
        rsa_matrices_listener = rsa_listener(raw, alpha)

        rsa_matrices_listener['ll'] = []
        rsa_matrices_listener['ll'] = rsa_matrices_speaker['ll']

        wd_scen, kl_scen, top_dist_scen = \
            comparison_rsa_listener(rsa_matrices_listener, scenarios_nouns[k], scenario_adjs[k],
                                    mturk_listener_scenarios[k], word_list, k, ent_limit, single_adj=single_adj)

        wd_lit.append(wd_scen['lit'])
        kl_lit.append(kl_scen['lit'])
        top_dist_lit.append(top_dist_scen['lit'])

    print("(S) Literal top 1 hit: {}".format(acc_func(top_dist_lit, 1)))
    print("(S) Literal top 3 hit: {}".format(acc_func(top_dist_lit, 3)))
    print("")
    print("(S) Literal JSD: {}".format(round(np.nanmean(kl_lit), 2)))
    print("")
    print("(S) Literal WD: {}".format(round(np.nanmean(wd_lit), 2)))

    return acc_func(top_dist_lit, 1), acc_func(top_dist_lit, 3), round(np.nanmean(kl_lit), 2), round(np.nanmean(wd_lit), 2)


def analyze_rsa(word_list, scenarios_nouns, scenario_adjs, adj_feats, mturk_scenarios, acc_func,
                rsa_dset_func, rsa_comparison_func, object_agg_func, ent_limit, alpha, results_dict, single_adj=False):
    '''
    :param word_list: list of nouns
    :param scenarios_nouns: list of scenarios nouns
    :param scenario_adjs: list of scenario adjectives
    :param adj_feats: noun adjectives similarities
    :param object_agg_func: aggregation function for combining word pair probabilities
    :param ent_limit: entropy upper bound
    :param alpha: pragmatic coefficient
    :param rsa_dset_func: function to compare data (rsa_listener or rsa_speaker)
    :param rsa_comparison_func: function to comparison_rsa_listener or comparison_rsa_speaker
    :param results_dict: dictionary to store results
    :return: [top 1 lit acc, top 1 prag acc, top 3 lit acc, top 3 prag acc, lit JSD, prag JSD, lit WD, prag WD]
    '''
    wd_lit = []
    wd_prag = []
    wd_uniform = []
    kl_lit = []
    kl_prag = []
    kl_uniform = []
    top_dist_lit = []
    top_dist_prag = []

    (mturk_dict, literal_dict, prag_dict, entropy_dict) = results_dict

    for k in range(len(scenario_adjs)):
        raw, _ = extract_scenario_table(scenario_adjs[k], scenarios_nouns[k], adj_feats, object_agg_func)
        rsa_matrices = rsa_dset_func(raw, alpha)

        mturk_dict[k] = {}
        literal_dict[k] = {}
        prag_dict[k] = {}
        entropy_dict[k] = {}

        if single_adj:
            wd_scen, kl_scen, top_dist_scen = \
                rsa_comparison_func(rsa_matrices, scenarios_nouns[k], scenario_adjs[k], mturk_scenarios[k], word_list, k,
                    ent_limit, results_dict=(mturk_dict[k], literal_dict[k], prag_dict[k], entropy_dict[k]), single_adj=single_adj)
        else:
            wd_scen, kl_scen, top_dist_scen = \
                rsa_comparison_func(rsa_matrices, scenarios_nouns[k], scenario_adjs[k], mturk_scenarios[k], word_list,
                                    k, ent_limit,
                                    results_dict=(mturk_dict[k], literal_dict[k], prag_dict[k], entropy_dict[k]))

        wd_lit.append(wd_scen['lit'])
        wd_prag.append(wd_scen['prag'])
        wd_uniform.append(wd_scen['uniform'])
        kl_lit.append(kl_scen['lit'])
        kl_prag.append(kl_scen['prag'])
        kl_uniform.append(kl_scen['uniform'])
        top_dist_lit.append(top_dist_scen['lit'])
        top_dist_prag.append(top_dist_scen['prag'])

    print("Literal top 1 hit: {}".format(acc_func(top_dist_lit, 1)))
    print("Pragmatic top 1 hit: {}".format(acc_func(top_dist_prag, 1)))

    print("Literal top 3 hit: {}".format(acc_func(top_dist_lit, 3)))
    print("Pragmatic top 3 hit: {}".format(acc_func(top_dist_prag, 3)))
    print("")
    print("Literal JSD: {}".format(round(np.nanmean(kl_lit), 2)))
    print("Pragmatic JSD: {}".format(round(np.nanmean(kl_prag), 2)))
    print("")
    print("Literal WD: {}".format(round(np.nanmean(wd_lit), 2)))
    print("Pragmatic WD: {}".format(round(np.nanmean(wd_prag), 2)))

    return acc_func(top_dist_lit, 1), acc_func(top_dist_prag, 1), acc_func(top_dist_lit, 3), acc_func(top_dist_prag, 3), \
           round(np.nanmean(kl_lit), 2), round(np.nanmean(kl_prag), 2), round(np.nanmean(wd_lit), 2), round(np.nanmean(wd_prag), 2)


def merge_scenarios(raw_file_str, match_arr_file=None):
    '''
    merge permutations of scenarios into one list
    :param raw_file_str: file name
    :param match_arr_file: only keep scenarios in match_arr
    :return: code and adjective scenarios merged in sequence
    '''
    with open(raw_file_str + '0.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios0 = scenarios['scenarios']

    with open(raw_file_str + '1.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios1 = scenarios['scenarios']

    with open(raw_file_str + '2.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios2 = scenarios['scenarios']
    scenarios = scenarios0 + scenarios1 + scenarios2

    filter_scenarios = []
    if match_arr_file:
        match_arr = np.load(match_arr_file)
        for i in range(len(scenarios)):
            if i%len(scenarios0) in match_arr:
                filter_scenarios.append(scenarios[i])
        scenarios = filter_scenarios[:]
    code_scenarios = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios = [adjs for ingroup, outgroup, adjs in scenarios]

    return code_scenarios, adj_scenarios


def get_count_accuracy_speaker(model_comp_list, model_comp_list_str, comparison='semantic'):
    '''
    top match accuracy of speaker and produce graphs if PLOT_PROB is True
    :param model_comp_list: list of (feat, comp)
    :param model_comp_list_str: names of features
    :param comparison: either semantic or pragmatic
    :return: None
    '''

    if comparison == 'semantic':
        _, speaker_data = load_semantic_scen(counts=True, filter=True)
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, speaker_qual_ratings = speaker_data
        tie_arr = np.asarray([0, 0, 0])
        model_count_prob_mat = np.zeros((len(mturk_speaker_scenarios), len(model_comp_list)))
        for i, scen in enumerate(mturk_speaker_scenarios):
            adj_set = adj_scenarios_speaker[i]
            noun_set = code_scenarios_speaker[i]
            model_choices = experiment_utils.abs_max_across_models(model_comp_list, adj_set, noun_set,
                                                        comp='speaker')

            print(" ")
            print("Scenario_p %d"%i)
            print("In group words: %s %s   outgroup words: %s"%(word_list[noun_set[0][0]], word_list[noun_set[0][1]],
                                                                word_list[noun_set[1][0]]))
            print("choices: ", adj_set)
            print("mturk choices count: ")
            for k in range(len(adj_set)):
                print(adj_set[k], mturk_speaker_scenarios[i][k])
            print("mturk avg. conf", speaker_qual_ratings[i])
            for j in range(len(model_comp_list)):
                model_count_prob_mat[i, j] = scen[model_choices[j]]
                print( model_comp_list_str[j], ": ", adj_set[model_choices[j]])

            model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='speaker')

            model_max = np.argmax(model_dist, axis=1)

            for j in range(len(model_comp_list)):
                match = np.where(model_dist[j] == model_dist[j, model_max[j]])
                if (len(match[0]) > 1):
                    tie_arr[j] += 1

            if PLOT_PROB:
                viz_utils.viz_raw_prob(i, adj_set, noun_set, word_list, model_comp_list, model_comp_list_str, 'speaker_semantic_geo')
                viz_utils.viz_scenario_speaker(i, model_dist, mturk_speaker_scenarios[i]/sum(mturk_speaker_scenarios[i]),
                                     model_comp_list_str, adj_set, [word_list[noun_set[0][0]], word_list[noun_set[0][1]],
                                                                     word_list[noun_set[1][0]]], 'semantic_geo')

        mturk_speaker_scenarios = np.asarray(mturk_speaker_scenarios)
        p_arr = model_count_prob_mat.sum(axis=0)/mturk_speaker_scenarios.sum()

        std_arr = [np.sqrt(p*(1-p)/mturk_speaker_scenarios.sum()) for p in p_arr]
        print(np.round(p_arr, 3))
        print(np.round(std_arr, 3))
        print("tie_arr", tie_arr)

    elif comparison == 'pragmatic':
    #PRAGMATIC
        _, speaker_data = load_pragmatic_scen(union=True)
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, speaker_qual_ratings = speaker_data

        alpha_arr = [0.1, 1.0, 5.0, 10.0]
        model_count_prob_mat = np.zeros((len(mturk_speaker_scenarios), len(alpha_arr)))
        lit_model_count = []
        overlap_count = np.zeros((len(alpha_arr),))
        tie_arr = np.asarray([0, 0])
        for i, scen in enumerate(mturk_speaker_scenarios):
            adj_set = adj_scenarios_speaker[i]
            noun_set = code_scenarios_speaker[i]

            model_choices = experiment_utils.abs_max_across_models([model_comp_list[0]], adj_set, noun_set,
                                                                   comp='speaker', idx=0)
            lit_choice = model_choices[0]
            lit_model_count.append(scen[model_choices[0]])

            print(" ")
            print("Scenario %d"%i)
            print("In group words: %s %s   outgroup words: %s"%(word_list[noun_set[0][0]], word_list[noun_set[0][1]],
                                                                word_list[noun_set[1][0]]))
            print("choices: ", adj_set)
            print("mturk choices count: ")
            for k in range(len(adj_set)):
                print(adj_set[k], mturk_speaker_scenarios[i][k])
            print("mturk avg. conf", speaker_qual_ratings[i])
            print("literal choice: ", adj_set[lit_choice])

            for j, alpha in enumerate(alpha_arr):
                model_choices = experiment_utils.abs_max_across_models([model_comp_list[1]], adj_set, noun_set,
                                                            comp='speaker', idx=0, alpha=alpha)
                prag_choice = model_choices[0]
                print("pragmatic choice: ", adj_set[prag_choice])
                if model_choices[0] == lit_choice:
                    overlap_count[j] += 1
                model_count_prob_mat[i, j] = scen[model_choices[0]]



            model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='speaker')
            model_max = np.argmax(model_dist, axis=1)
            print(model_dist)
            for j in range(len(model_comp_list)):
                match = np.where(model_dist[j] == model_dist[j, model_max[j]])
                if (len(match[0]) > 1):
                    tie_arr[j] += 1


            if PLOT_PROB:
                rsa_mat = experiment_utils.speaker_choice_given_M_D(model_comp_list[0][0], adj_set, noun_set, full_mat=True)
                viz_utils.viz_prag_prob(i, adj_set, noun_set, word_list, rsa_mat, ['raw', 'ls', 'll', 'ps'], comp='speaker')
                model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='speaker')
                viz_utils.viz_scenario_speaker(i, model_dist, mturk_speaker_scenarios[i] / sum(mturk_speaker_scenarios[i]),
                                 ['literal', 'pragmatic'], adj_set, [word_list[noun_set[0][0]], word_list[noun_set[0][1]],
                                                                word_list[noun_set[1][0]]], 'pragmatic_geo')

        lit_model_count = np.asarray(lit_model_count).reshape(-1, 1)
        model_count_prob_mat = np.concatenate((lit_model_count, model_count_prob_mat), axis=1)
        p_arr = model_count_prob_mat.sum(axis=0) / np.sum(mturk_speaker_scenarios)

        std_arr = [np.sqrt(p * (1 - p) / np.sum(mturk_speaker_scenarios)) for p in p_arr]
        print(np.round(p_arr, 3))
        print(np.round(std_arr, 3))
        print(tie_arr)


def get_count_accuracy_listener(model_comp_list, model_comp_list_str, comparison='semantic'):
    '''

    top match accuracy of listener and produce graphs if PLOT_PROB is True
    :param model_comp_list: list of (feat, comp)
    :param model_comp_list_str: names of features
    :param comparison: either semantic or pragmatic
    '''

    if comparison == 'semantic':
        listener_data, _ = load_semantic_scen(counts=True, filter=True)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, listener_qual_ratings = listener_data
        tie_arr = np.asarray([0, 0, 0])
        model_count_prob_mat = np.zeros((len(mturk_listener_scenarios), len(model_comp_list)))
        print("num_scen: ", len(mturk_listener_scenarios))
        for i, scen in enumerate(mturk_listener_scenarios):
            adj_set = adj_scenarios_listener[i]
            noun_set = code_scenarios_listener[i]
            model_choices = experiment_utils.abs_max_across_models(model_comp_list, adj_set, noun_set,
                                                        comp='listener', idx=0)

            noun_pairs = list(itertools.combinations([noun_set[0][0], noun_set[0][1], noun_set[1][0]], 2))
            np_words = [(word_list[np[0]], word_list[np[1]]) for np in noun_pairs]

            print(" ")
            print("Scenario %d" % i)
            print("possible codenames: ", word_list[noun_set[0][0]], word_list[noun_set[0][1]], word_list[noun_set[1][0]])
            print("all adjectives: ", adj_set)
            print("given adjective clue: ", adj_set[0])
            print("mturk choices count: ")
            for k in range(len(np_words)):
                print(np_words[k], mturk_listener_scenarios[i][0][k])
            print("mturk avg. conf", listener_qual_ratings[i])


            for j in range(len(model_comp_list)):
                model_count_prob_mat[i, j] = scen[0][model_choices[j]]
                print(model_comp_list_str[j], (word_list[noun_pairs[model_choices[j]][0]], word_list[noun_pairs[model_choices[j]][1]]))


            model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='listener', idx=0)
            model_max = np.argmax(model_dist, axis=1)

            for j in range(len(model_comp_list)):
                match = np.where(model_dist[j] == model_dist[j, model_max[j]])
                if (len(match[0]) > 1):
                    tie_arr[j] += 1

            if PLOT_PROB:
                viz_utils.viz_raw_prob(i, adj_set, noun_set, word_list, model_comp_list, model_comp_list_str, 'listener_semantic')
                viz_utils.viz_scenario_listener(i, model_dist, mturk_listener_scenarios[i][0] / sum(mturk_listener_scenarios[i][0]),
                                  model_comp_list_str, adj_set, np_words, 'semantic_geo')
        print("tie_arr", tie_arr)
        p_arr = model_count_prob_mat.sum(axis=0)/mturk_listener_scenarios.sum()

        std_arr = [np.sqrt(p * (1 - p) / mturk_listener_scenarios.sum()) for p in p_arr]
        print(np.round(p_arr, 3))
        print(np.round(std_arr, 3))


    elif comparison == 'pragmatic':
        listener_data, _ = load_pragmatic_scen(union=True)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, listener_qual_ratings = listener_data

        alpha_arr = [0.1, 1.0, 5.0, 10.0]
        model_count_prob_mat = np.zeros((len(mturk_listener_scenarios), len(alpha_arr)))
        lit_model_count = []
        overlap_count = np.zeros((len(alpha_arr),))
        tie_arr = np.asarray([0, 0])
        for i, scen in enumerate(mturk_listener_scenarios):

            adj_set = adj_scenarios_listener[i]
            noun_set = code_scenarios_listener[i]

            noun_pairs = list(itertools.combinations([noun_set[0][0], noun_set[0][1], noun_set[1][0]], 2))

            model_choices = experiment_utils.abs_max_across_models([model_comp_list[0]], adj_set, noun_set,
                                                                   comp='listener', idx=0)
            lit_model_count.append(scen[0][model_choices[0]])
            lit_choice = model_choices[0]
            np_words = [(word_list[np[0]], word_list[np[1]]) for np in noun_pairs]

            # print info
            print(" ")
            print("Scenario %d" % i)
            print("possible codenames: ", word_list[noun_set[0][0]], word_list[noun_set[0][1]], word_list[noun_set[1][0]])
            print("all adjectives: ", adj_set)
            print("given adjective clue: ", adj_set[0])
            print("mturk choices count: ")
            for k in range(len(np_words)):
                print(np_words[k], mturk_listener_scenarios[i][0][k])
            print("mturk avg. conf", listener_qual_ratings[i])
            print("literal choice: ", word_list[noun_pairs[lit_choice][0]] + ' ' + word_list[noun_pairs[lit_choice][1]])

            for j, alpha in enumerate(alpha_arr):
                model_choices = experiment_utils.abs_max_across_models([model_comp_list[1]], adj_set, noun_set,
                                                            comp='listener', idx=0, alpha=alpha)

                prag_choice = model_choices[0]
                model_count_prob_mat[i, j] = scen[0][prag_choice]

                print("prag choice: ",  word_list[noun_pairs[prag_choice][0]] + ' ' + word_list[noun_pairs[prag_choice][1]])
                if model_choices[0] == lit_choice:
                    overlap_count[j] += 1

                model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='listener')
                model_max = np.argmax(model_dist, axis=1)

                for j in range(len(model_comp_list)):
                    match = np.where(model_dist[j] == model_dist[j, model_max[j]])
                    if (len(match[0]) > 1):
                        tie_arr[j] += 1

            if PLOT_PROB:
                rsa_mat = experiment_utils.listener_choice_given_M_D(model_comp_list[0][0], adj_set, noun_set, idx=0, full_mat=True)
                viz_utils.viz_prag_prob(i, adj_set, noun_set, word_list, rsa_mat, ['raw', 'll', 'ls', 'pl'])
                model_dist = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, comp='listener')
                print(mturk_listener_scenarios[i][0] / sum(mturk_listener_scenarios[i][0]))
                viz_utils.viz_scenario_listener(i, model_dist, mturk_listener_scenarios[i][0] / sum(mturk_listener_scenarios[i][0]),
                                 ['Bigram Literal', 'Bigram Pragmatic'], adj_set, np_words, 'pragmatic_geo')


        lit_model_count = np.asarray(lit_model_count).reshape(-1, 1)
        model_count_prob_mat = np.concatenate((lit_model_count, model_count_prob_mat), axis=1)
        p_arr = model_count_prob_mat.sum(axis=0) / mturk_listener_scenarios.sum()

        std_arr = [np.sqrt(p * (1 - p) / mturk_listener_scenarios.sum()) for p in p_arr]
        print(np.round(p_arr, 3))
        print(np.round(std_arr, 3))
        print("tie arr", tie_arr)


def run_rsa_analysis(model_feature_list, model_feature_names, csv_keys, word_list, ex=3):
    '''
    run semantic top match rsa analysis and save results in ../results/temp/
    :param model_feature_list: list of (adj_noun_feats, comp_func)
    :param model_feature_names: names of input features
    :param ex: experiment number (1, 2 or 3)
    :return: None
    '''
    if ex == 1:
        listener_data, speaker_data = load_full_set_scen()
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios = speaker_data
    elif ex == 2:
        listener_data, speaker_data = load_semantic_scen_ex2(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data
    else:
        listener_data, speaker_data = load_semantic_scen(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    print("speaker", len(code_scenarios_speaker))
    print("listener", len(code_scenarios_listener))

    entropy_listener = 10
    entropy_speaker = 10

    data_set_str = 'semantic'

    csv_file = open('../results/temp/' + data_set_str + '_summary.csv', 'a')
    results_writer = csv.writer(csv_file)
    results_writer.writerow(csv_keys)

    object_agg_func = np.prod
    prag_acc = []
    for i, adj_feats in enumerate(model_feature_list):
        alpha = 1.0
        print(model_feature_names[i])

        SIM_TYPE = model_feature_names[i] + '_' + str(alpha) + object_agg_func.__name__.upper()
        print('ALPHA: ' + str(alpha) + ' - ' + object_agg_func.__name__.upper())

        listener_arr, listener_literal_arr = analyze_listener(mturk_listener_scenarios, word_list, code_scenarios_listener,
                                                             adj_scenarios_listener, adj_feats, object_agg_func,
                                                             entropy_listener, alpha, single_adj=True)

        speaker_arr, speaker_literal_arr = analyze_speaker(mturk_speaker_scenarios, word_list, code_scenarios_speaker,
                                                           adj_scenarios_speaker, adj_feats, object_agg_func,
                                                           entropy_speaker, alpha=alpha)

        merged_arr = listener_arr[:8] + listener_literal_arr[:4] + speaker_arr[:8] + speaker_literal_arr[:4]
        prag_acc.append(listener_arr[2])
        merged_arr = [SIM_TYPE] + merged_arr
        results_writer.writerow(merged_arr)
        print("\n")
    csv_file.close()


def run_rsa_analysis_pragmatic(model_feature_list, model_feature_names, csv_keys, word_list):
    '''
    run pragmatic top match rsa analysis and save results in ../results/temp/
    :param model_feature_list: list of (adj_noun_feats, comp_func)
    :param model_feature_names: names of input features
    :return: None
    '''

    listener_data, speaker_data = load_pragmatic_scen()
    code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
    code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    print("speaker", len(code_scenarios_speaker))
    print("listener", len(code_scenarios_listener))

    entropy_listener = 10
    entropy_speaker = 10

    data_set_str = 'pragmatic'

    csv_file = open('../results/temp/' + data_set_str + '_summary.csv', 'a')
    results_writer = csv.writer(csv_file)
    results_writer.writerow(csv_keys)

    object_agg_func = np.prod

    # for object_agg_func in [np.max, np.mean, np.prod]:
    prag_acc = []
    for i, adj_feats in enumerate(model_feature_list):
        for alpha in [0.1, 1.0, 5.0]:
            print(model_feature_names[i])

            SIM_TYPE = model_feature_names[i] + '_' + str(alpha) + object_agg_func.__name__.upper()
            print('ALPHA: ' + str(alpha) + ' - ' + object_agg_func.__name__.upper())

            listener_arr, listener_literal_arr = analyze_listener(mturk_listener_scenarios, word_list,
                                                                  code_scenarios_listener,
                                                                  adj_scenarios_listener, adj_feats, object_agg_func,
                                                                  entropy_listener, alpha, single_adj=True)

            speaker_arr, speaker_literal_arr = analyze_speaker(mturk_speaker_scenarios, word_list,
                                                               code_scenarios_speaker,
                                                               adj_scenarios_speaker, adj_feats, object_agg_func,
                                                               entropy_speaker, alpha=alpha)

            merged_arr = listener_arr[:8] + listener_literal_arr[:4] + speaker_arr[:8] + speaker_literal_arr[:4]
            prag_acc.append(listener_arr[2])
            merged_arr = [SIM_TYPE] + merged_arr
            results_writer.writerow(merged_arr)

        print("\n")
    #
    csv_file.close()

def gen_speaker_stats(model_comp_list, model_list_str, exp,  metric="top"):
    '''
    generate speaker statistical analysis form in ../stats_test/
    :param model_comp_list: list of (adj_noun_feats, comp_func)
    :param model_list_str: names of input features
    :param exp: experiment number (1, 2 or 3)
    :param metric: top match or correlation
    :return: None
    '''
    if exp == 1:
        test_listener, test_speaker = load_full_set_scen()
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios = test_speaker
    elif exp == 2:
        listener_data, speaker_data = load_semantic_scen_ex2(counts=False)
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data
    elif exp == 3:
        listener_data, speaker_data = load_semantic_scen(counts=False)
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    else:
        listener_data, speaker_data = load_pragmatic_scen(counts=False)
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    csv_keys = ["metric", "conf_id", "score"]
    csv_file = open("../stat_test/ex" + str(exp) +"_".join(model_list_str) + '_speaker_' +  metric + ".csv", 'w')
    results_writer = csv.writer(csv_file)
    results_writer.writerow(csv_keys)

    min_arr = []
    for feat, comp in model_comp_list:
        flat_arr = []
        for adj in feat.keys():
            flat_arr += list(feat[adj])
        min = np.min(flat_arr)
        min_arr.append(min)

    for i in range(len(mturk_speaker_scenarios)):

        adj_set = adj_scenarios_speaker[i]
        noun_set = code_scenarios_speaker[i]
        norm_prob_arr = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, idx=0, comp='speaker')
        max_ind = np.argmax(mturk_speaker_scenarios[i])
        for j in range(len(model_comp_list)):

            if metric == "top":
                if len(np.where(mturk_speaker_scenarios[i] == mturk_speaker_scenarios[i][max_ind])[0]) > 1:
                    score = np.nan
                elif np.argmax(norm_prob_arr[j]) == max_ind:
                    score = 1
                else:
                    score = 0
            else:
                score = scipy.stats.spearmanr(mturk_speaker_scenarios[i], norm_prob_arr[j])[0]
            results_writer.writerow([model_list_str[j], i, score])
    csv_file.close()


def gen_listener_stats(model_comp_list, model_list_str, exp,  metric="top"):
    '''
    generate listener statistical analysis form in ../stats_test/
    :param model_comp_list: list of (adj_noun_feats, comp_func)
    :param model_list_str: names of input features
    :param exp: experiment number (1, 2 or 3)
    :param metric: top match or correlation
    :return: None
    '''
    if exp == 1:
        test_listener, _ = load_full_set_scen()
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios = test_listener
    elif exp == 2:
        listener_data, _ = load_semantic_scen_ex2(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
    elif exp == 3:
        listener_data, _ = load_semantic_scen(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data

    else:
        listener_data, _ = load_pragmatic_scen(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data

    csv_keys = ["metric", "conf_id", "score"]
    csv_file = open("../stat_test/ex" + str(exp) + "_".join(model_list_str) + '_listener_' + metric + ".csv", 'w')
    results_writer = csv.writer(csv_file)
    results_writer.writerow(csv_keys)

    min_arr = []
    for feat, comp in model_comp_list:
        flat_arr = []
        for adj in feat.keys():
            flat_arr += list(feat[adj])
        min = np.min(flat_arr)
        min_arr.append(min)

    for i in range(len(mturk_listener_scenarios)):
        adj_set = adj_scenarios_listener[i]
        noun_set = code_scenarios_listener[i]
        norm_prob_arr = experiment_utils.dist_across_models(model_comp_list, adj_set, noun_set, idx=0, comp='listener')

        max_ind = np.argmax(mturk_listener_scenarios[i])
        for j in range(len(model_comp_list)):
            if metric == "top":
                if len(np.where(mturk_listener_scenarios[i][0] == mturk_listener_scenarios[i][0][max_ind])[0]) > 1:
                    score = np.nan
                elif len(np.where(norm_prob_arr[j] == max(norm_prob_arr[j]))[0]) > 1:
                    print("tied: ", i, model_comp_list_str[j])
                elif np.argmax(norm_prob_arr[j]) == max_ind:
                    score = 1
                else:
                    score = 0
            else:
                try:
                    score = scipy.stats.spearmanr(mturk_listener_scenarios[i][0], norm_prob_arr[j])[0]
                except:
                    score = 0
            results_writer.writerow([model_list_str[j], i, score])

    csv_file.close()


def run_model_probability(model_list, kl=False, rank_corr=False, likelihood=False, sparse=False, include_prag=False, ex=3):
    '''
    compare semantic model kl, rank correlation, likelihood
    :param model_list: list of (adj_noun_feats, comp_func)
    :param kl: True to compute kl divergence
    :param rank_corr: True to compute rank corr
    :param likelihood: True to compute likelihood
    :param sparse: True to compute sparse
    :param include_prag: include pragmatic features
    :param ex: experiment number (1, 2, 3)
    :return: None
    '''
    if ex == 1:
        listener_data, speaker_data = load_full_set_scen()
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios = speaker_data
    elif ex == 2:
        listener_data, speaker_data = load_semantic_scen_ex2(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data
    else:
        listener_data, speaker_data = load_semantic_scen(counts=False)
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    model_comp_list = [(model, 'ls') for (model, comp) in model_list]
    if include_prag:
        for model in model_list:
            model_comp_list.append((model, 'ps'))

    scenario_prob = np.zeros((len(mturk_speaker_scenarios), len(model_comp_list)))
    min_arr = []
    for feat, comp in model_comp_list:
        flat_arr =[]
        for adj in feat.keys():
            flat_arr += list(feat[adj])
        min = np.min(flat_arr)
        min_arr.append(min)

    sparse_count = np.zeros((len(model_comp_list)))
    for i in range(len(mturk_speaker_scenarios)):
        adj_set = adj_scenarios_speaker[i]
        noun_set = code_scenarios_speaker[i]
        norm_prob_arr = experiment_utils.model_prob_given_choice(model_comp_list,
                                                                 adj_set, noun_set,
                                                                 experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                 comp='speaker')
        max_ind = np.argmax(mturk_speaker_scenarios[i])

        #kl divergence
        if kl:
            for j in range(len(model_comp_list)):
                kl = scipy.stats.entropy(mturk_speaker_scenarios[i], norm_prob_arr[j, :])
                scenario_prob[i, j] = kl
        elif rank_corr:
            for j in range(len(model_comp_list)):
                corr = scipy.stats.spearmanr(mturk_speaker_scenarios[i], norm_prob_arr[j, :])
                scenario_prob[i, j] = corr[0] #if corr[0] > 0 else 0
        elif likelihood:
            for j in range(len(model_comp_list)):
                log_prob = np.log(norm_prob_arr[j, :])
                log_scen = np.dot(mturk_speaker_scenarios[i], log_prob)
                scenario_prob[i, j] = 1/np.exp(log_scen)
            scenario_prob[i, :] /= np.sum(scenario_prob[i, :])
        elif sparse:
            for j in range(len(model_comp_list)):
                raw = raw_scenario_table(adj_set, noun_set[0] + noun_set[1], model_comp_list[j])
                speaker_raw = raw[:, :2]
                match_arr = np.where(speaker_raw.flatten() == min_arr[j])
                #if len(match_arr[0]) > 0:
                sparse_count[j] += len(match_arr[0])

        else:
            norm_prob_arr /= np.sum(norm_prob_arr, axis=0)
            scenario_prob[i, :] = norm_prob_arr[:, max_ind]


    print("Speaker: %d config"%len(code_scenarios_speaker))
    for m in scenario_prob.mean(axis=0):
        print(m)
    for e in scipy.stats.sem(scenario_prob):
        print(e)

    print(np.asarray(sparse_count)/len(scenario_prob))
    print(sparse_count)
    print(scipy.stats.ttest_ind(scenario_prob[:, 0], scenario_prob[:, 1]))


    sparse_count = np.zeros((len(model_comp_list)))
    model_comp_list = [(model, 'll') for (model, comp) in model_list]
    if include_prag:
        for model in model_list:
            model_comp_list.append((model, 'pl'))

    scenario_prob = np.zeros((len(mturk_listener_scenarios), len(model_comp_list)))
    for i in range(len(mturk_listener_scenarios)):
        adj_set = adj_scenarios_listener[i]
        noun_set = code_scenarios_listener[i]
        norm_prob_arr = experiment_utils.model_prob_given_choice(model_comp_list, adj_set, noun_set,
                                                                 experiment_utils.listener_choice_given_M_D,
                                                                 comp='listener', idx=0)
        max_ind = np.argmax(mturk_listener_scenarios[i][0])

        #kl divergence
        if kl:
            for j in range(len(model_comp_list)):
                kl = scipy.stats.entropy(mturk_listener_scenarios[i][0], norm_prob_arr[j,:])
                scenario_prob[i, j] = kl
        elif rank_corr:
            for j in range(len(model_comp_list)):
                try:
                    corr = scipy.stats.spearmanr(mturk_listener_scenarios[i][0], norm_prob_arr[j,:])
                    scenario_prob[i, j] = corr[0] #if corr[0] > 0 else 0
                except:
                    print("error")
                    print(mturk_listener_scenarios[i][0], norm_prob_arr[j,:])

        elif likelihood:
            for j in range(len(model_comp_list)):
                log_prob = np.log(norm_prob_arr[j, :])
                log_scen = np.dot(mturk_speaker_scenarios[i], log_prob)
                scenario_prob[i, j] = 1/np.exp(log_scen)
            scenario_prob[i, :] /= np.sum(scenario_prob[i, :])
        elif sparse:
            for j in range(len(model_comp_list)):
                raw = raw_scenario_table(adj_set, noun_set[0] + noun_set[1], model_comp_list[j])
                speaker_raw = raw[0, :]
                match_arr = np.where(speaker_raw.flatten() == min_arr[j])
                #if len(match_arr[0]) > 0:
                sparse_count[j] += len(match_arr[0])
        else:
            norm_prob_arr /= np.sum(norm_prob_arr, axis=0)
            scenario_prob[i, :] = norm_prob_arr[:, max_ind]

    print("Listener: %d config"%len(scenario_prob))
    for m in scenario_prob.mean(axis=0):
        print(m)
    for e in scipy.stats.sem(scenario_prob):
        print(e)
    print(sparse_count)
    print(np.asarray(sparse_count)/len(scenario_prob))
    print(scipy.stats.ttest_ind(scenario_prob[:, 0], scenario_prob[:, 1]))


def run_model_probability_prag(model_comp_list, kl=False, rank_corr=False, likelihood=False):
    '''
    compare pragmatic kl, rank correlation, likelihood
    :param kl: True to compute kl divergence
    :param rank_corr: True to compute rank corr
    :param likelihood: True to compute likelihood
    :return: None
    '''

    listener_data, speaker_data = load_pragmatic_scen(counts=True)
    code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
    code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data

    alpha_arr = [0.1, 1.0, 5.0]
    model_count_prob_mat = np.zeros((len(mturk_speaker_scenarios), len(alpha_arr)))
    lit_model_prob = np.zeros((len(mturk_speaker_scenarios), 1))
    print(len(mturk_speaker_scenarios))
    print(len(mturk_listener_scenarios))

    for i in range(len(mturk_speaker_scenarios)):
        adj_set = adj_scenarios_speaker[i]
        noun_set = code_scenarios_speaker[i]
        lit_prob = experiment_utils.model_prob_given_choice([model_comp_list[0]],
                                                                 adj_set, noun_set,
                                                                 experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                 comp='speaker')
        max_ind = np.argmax(mturk_speaker_scenarios[i])


        #kl divergence
        if kl:
            lit_model_prob[i] = scipy.stats.entropy(mturk_speaker_scenarios[i], lit_prob[0])
            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]],
                                                                 adj_set, noun_set,
                                                                 experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                 comp='speaker', alpha=alpha)
                kl = scipy.stats.entropy(mturk_speaker_scenarios[i], prag_prob[0])
                model_count_prob_mat[i, j] = kl

        elif rank_corr:
            lit_model_prob[i] = scipy.stats.spearmanr(mturk_speaker_scenarios[i], lit_prob[0])[0]
            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]],
                                                                     adj_set, noun_set,
                                                                     experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                     comp='speaker', alpha=alpha)
                corr = scipy.stats.spearmanr(mturk_speaker_scenarios[i], prag_prob[0])
                model_count_prob_mat[i, j] = corr[0] #if corr[0] > 0 else 0
        elif likelihood:
            log_prob = np.log(lit_prob[0])
            log_scen = np.dot(mturk_speaker_scenarios[i], log_prob)
            lit_model_prob[i] = np.exp(log_scen)

            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]],
                                                                     adj_set, noun_set,
                                                                     experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                     comp='speaker', alpha=alpha)
                log_prob = np.log(prag_prob[0])
                log_scen = np.dot(mturk_speaker_scenarios[i], log_prob)
                try:
                    model_count_prob_mat[i, j] = np.exp(log_scen)
                except RuntimeWarning:
                    print("warning 0 encountered")
                    print(i, j)
                    model_count_prob_mat[i, j] = 0

            row_sum = model_count_prob_mat[i].sum() + lit_model_prob[i]
            model_count_prob_mat[i, :] /= row_sum
            lit_model_prob[i] /= row_sum
        else:
            lit_model_prob[i] = lit_prob[0][max_ind]
            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]],
                                                                     adj_set, noun_set,
                                                                     experiment_utils.speaker_choice_given_M_D, idx=0,
                                                                     comp='speaker', alpha=alpha)
                model_count_prob_mat[i, j] = prag_prob[0][max_ind] #if corr[0] > 0 else 0
            row_sum = model_count_prob_mat[i].sum() + lit_model_prob[i]
            model_count_prob_mat[i] /= row_sum
            lit_model_prob[i] /= row_sum

    print("Speaker")
    model_count_prob_mat = np.concatenate((lit_model_prob, model_count_prob_mat), axis=1)
    print(model_count_prob_mat.mean(axis=0))
    print(scipy.stats.sem(model_count_prob_mat))
    model_comp_list = [(adj_feats, prag.rstrip('s') + 'l') for adj_feats, prag in model_comp_list]

    model_count_prob_mat = np.zeros((len(mturk_listener_scenarios), len(alpha_arr)))
    lit_model_prob = np.zeros((len(mturk_listener_scenarios), 1))

    for i in range(len(mturk_listener_scenarios)):
        adj_set = adj_scenarios_listener[i]
        noun_set = code_scenarios_listener[i]
        lit_prob = experiment_utils.model_prob_given_choice([model_comp_list[0]], adj_set, noun_set,
                                                                 experiment_utils.listener_choice_given_M_D,
                                                                 comp='listener', idx=0)
        max_ind = np.argmax(mturk_listener_scenarios[i][0])

        #kl divergence
        if kl:
            lit_model_prob[i] = scipy.stats.entropy(mturk_listener_scenarios[i][0], lit_prob[0])
            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]], adj_set, noun_set,
                                                                 experiment_utils.listener_choice_given_M_D,
                                                                 comp='listener', idx=0, alpha=alpha)
                kl = scipy.stats.entropy(mturk_listener_scenarios[i][0], prag_prob[0])
                model_count_prob_mat[i, j] = kl

        elif rank_corr:
            lit_model_prob[i] = scipy.stats.spearmanr(mturk_listener_scenarios[i][0], lit_prob[0])[0]

            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]], adj_set, noun_set,
                                                                     experiment_utils.listener_choice_given_M_D,
                                                                     comp='listener', idx=0, alpha=alpha)
                corr = scipy.stats.spearmanr(mturk_listener_scenarios[i][0], prag_prob[0])
                model_count_prob_mat[i, j] = corr[0]  # if corr[0] > 0 else 0

        elif likelihood:
            log_prob = np.log(lit_prob[0])
            log_scen = np.dot(mturk_listener_scenarios[i], log_prob)
            lit_model_prob[i] = np.exp(log_scen)

            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]], adj_set, noun_set,
                                                                     experiment_utils.listener_choice_given_M_D,
                                                                     comp='listener', idx=0, alpha=alpha)
                log_prob = np.log(prag_prob[0])
                log_scen = np.dot(mturk_listener_scenarios[i], log_prob)
                try:
                    model_count_prob_mat[i, j] = np.exp(log_scen)
                except RuntimeWarning:
                    print("warning 0 encountered")
                    print(i, j)
                    model_count_prob_mat[i, j] = 0

            row_sum = model_count_prob_mat[i].sum() + lit_model_prob[i]
            model_count_prob_mat[i, :] /= row_sum
            lit_model_prob[i] /= row_sum

        else:
            lit_model_prob[i] = lit_prob[0][max_ind]
            for j, alpha in enumerate(alpha_arr):
                prag_prob = experiment_utils.model_prob_given_choice([model_comp_list[1]], adj_set, noun_set,
                                                                     experiment_utils.listener_choice_given_M_D,
                                                                     comp='listener', idx=0, alpha=alpha)
                model_count_prob_mat[i, j] = prag_prob[0][max_ind] #if corr[0] > 0 else 0
            row_sum = model_count_prob_mat[i].sum() + lit_model_prob[i]
            model_count_prob_mat[i] /= row_sum
            lit_model_prob[i] /= row_sum

    print("Listener")
    model_count_prob_mat = np.concatenate((lit_model_prob, model_count_prob_mat), axis=1)
    print(model_count_prob_mat.mean(axis=0))
    print(scipy.stats.sem(model_count_prob_mat))


def load_pragmatic_scen(union=True, counts=False):
    '''
    helper function to load pragmatic experiment data
    :param union: only include top rater confidence scenarios for both listener and speaker
    :param counts: return as counts
    :return: (codename listener scenarios, adjective listener scenarios, mturk listener data, mturk listener confidence),
            (codename speaker scenarios, adjective speaker scenarios, mturk speaker data, mturk speaker confidence)
    '''
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'
    word_list, adj_list, _, _, _ = data_utils.load_pickle_files(file_path, codenames_file, 'adj_choosen.pickle')

    code_scenarios_listener, adj_scenarios_listener = merge_scenarios(
        '../output/gibbs_pragmatic_listener100000_used')

    mturk_listener_scenarios, listener_qual_ratings = data_utils.merge_listener_with_ratings(code_scenarios_listener, (
        '../data/listener_pragmatic3.csv', 8), ('../data/listener_pragmatic_filter3.csv', 8), word_list,
                                                                                             '../../graphics_gen/pragmatic_listener3.npy',
                                                                                             num_perm=3, scen_len=60, counts=counts,
                                                                                             match_arr_file='../../graphics_gen/listener_pragmatic_match.npy')
    code_scenarios_listener, adj_scenarios_listener = merge_scenarios(
        '../output/gibbs_pragmatic_listener100000_conf',
        match_arr_file='../../graphics_gen/listener_pragmatic_match.npy')

    if union:
        code_scenarios_listener_union, adj_scenarios_listener_union = merge_scenarios(
            '../output/gibbs_pragmatic_listener100000_union')

        mturk_listener_scenarios_union, listener_qual_ratings_union = data_utils.parse_listener_with_ratings(
            code_scenarios_listener_union, ('../data/listener_pragmatic_union.csv', 8), word_list, counts=counts)

        code_scenarios_listener = code_scenarios_listener + code_scenarios_listener_union
        adj_scenarios_listener = adj_scenarios_listener + adj_scenarios_listener_union

        mturk_listener_scenarios = np.concatenate((mturk_listener_scenarios, mturk_listener_scenarios_union), axis=0)
        listener_qual_ratings = np.concatenate((listener_qual_ratings, listener_qual_ratings_union), axis=0)

    code_scenarios_speaker, adj_scenarios_speaker = merge_scenarios('../output/gibbs_pragmatic_speaker100000_used')

    mturk_speaker_scenarios, speaker_qual_ratings = data_utils.merge_speaker_with_ratings(adj_scenarios_speaker, (
        '../data/speaker_pragmatic3.csv', 9), ('../data/speaker_pragmatic_filter3.csv', 9),
                                                                                          '../../graphics_gen/pragmatic_speaker3.npy',
                                                                                          num_perm=3, scen_len=60, counts=counts,
                                                                                          match_arr_file='../../graphics_gen/speaker_pragmatic_match.npy')

    code_scenarios_speaker, adj_scenarios_speaker = merge_scenarios('../output/gibbs_pragmatic_speaker100000_conf',
                                                                    match_arr_file='../../graphics_gen/speaker_pragmatic_match.npy')

    if union:
        code_scenarios_speaker_union, adj_scenarios_speaker_union = merge_scenarios(
            '../output/gibbs_pragmatic_speaker100000_union')

        mturk_speaker_scenarios_union, speaker_qual_ratings_union = data_utils.parse_speaker_with_ratings(
            ('../data/speaker_pragmatic_union.csv', 9), adj_scenarios_speaker_union, counts=counts)

        code_scenarios_speaker = code_scenarios_speaker + code_scenarios_speaker_union
        adj_scenarios_speaker = adj_scenarios_speaker + adj_scenarios_speaker_union

        mturk_speaker_scenarios = np.concatenate((mturk_speaker_scenarios, mturk_speaker_scenarios_union), axis=0)
        speaker_qual_ratings = np.concatenate((speaker_qual_ratings, speaker_qual_ratings_union), axis=0)

    return (code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, listener_qual_ratings), \
           (code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, speaker_qual_ratings)


def load_semantic_scen(counts=False, filter=True):
    '''
    helper function to load semantic experiment data
    :param filter: only include confidence filtered scenarios
    :param counts: return as counts
    :return: (codename listener scenarios, adjective listener scenarios, mturk listener data, mturk listener confidence),
            (codename speaker scenarios, adjective speaker scenarios, mturk speaker data, mturk speaker confidence)
    '''
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'
    word_list, adj_list, _, _, _ = data_utils.load_pickle_files(file_path, codenames_file, 'adj_choosen.pickle')

    code_scenarios_listener, adj_scenarios_listener = merge_scenarios(
        '../output/gibbs_semantic_geo3_listener100000_used')

    code_scenarios_speaker, adj_scenarios_speaker = merge_scenarios('../output/gibbs_semantic_geo3_speaker100000_used')

    if filter:
        mturk_listener_scenarios, listener_qual_ratings = data_utils.merge_listener_with_ratings(code_scenarios_listener, (
            '../data/listener_semantic3.csv', 9), ('../data/listener_semantic_filter3.csv', 8), word_list,
            '../../graphics_gen/semantic_listener3.npy', num_perm=3, counts=counts, match_arr_file='../../graphics_gen/listener_semantic_match.npy')

        code_scenarios_listener, adj_scenarios_listener = merge_scenarios(
            '../output/gibbs_semantic_geo3_listener100000_conf',
            match_arr_file='../../graphics_gen/listener_semantic_match.npy')

        mturk_speaker_scenarios, speaker_qual_ratings = data_utils.merge_speaker_with_ratings(adj_scenarios_speaker, (
            '../data/speaker_semantic3.csv', 9), ('../data/speaker_semantic_filter3.csv', 10),
            '../../graphics_gen/semantic_speaker3.npy',
            counts=counts, num_perm=3,
            match_arr_file='../../graphics_gen/speaker_semantic_match.npy')

        code_scenarios_speaker, adj_scenarios_speaker = merge_scenarios(
            '../output/gibbs_semantic_geo3_speaker100000_conf',
            match_arr_file='../../graphics_gen/speaker_semantic_match.npy')

    else:
        mturk_listener_scenarios, listener_qual_ratings = data_utils.merge_listener_with_ratings(
            code_scenarios_listener, (
                '../data/listener_semantic3.csv', 9), ('../data/listener_semantic_filter3.csv', 8), word_list,
            '../../graphics_gen/semantic_listener3.npy', num_perm=3, counts=counts)

        code_scenarios_listener, adj_scenarios_listener = merge_scenarios(
            '../output/gibbs_semantic_geo3_listener100000_conf')

        mturk_speaker_scenarios, speaker_qual_ratings = data_utils.merge_speaker_with_ratings(adj_scenarios_speaker, (
            '../data/speaker_semantic3.csv', 9), ('../data/speaker_semantic_filter3.csv', 10),
             '../../graphics_gen/semantic_speaker3.npy', counts=counts, num_perm=3)

        code_scenarios_speaker, adj_scenarios_speaker = merge_scenarios(
            '../output/gibbs_semantic_geo3_speaker100000_conf')

    return (code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, listener_qual_ratings), \
           (code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, speaker_qual_ratings)


def load_semantic_scen_ex2(counts=False):
    '''
    helper function to load semantic 3 noun 4 adj experiment data
    :param counts: return as counts
    :return: (codename listener scenarios, adjective listener scenarios, mturk listener data, mturk listener confidence),
            (codename speaker scenarios, adjective speaker scenarios, mturk speaker data, mturk speaker confidence)
    '''
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'
    word_list, adj_list, _, _, _ = data_utils.load_pickle_files(file_path, codenames_file, 'adj_choosen.pickle')

    with open('../output/gibbs_uniform_listener100000_used.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']
    code_scenarios_listener = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios_listener = [adjs for ingroup, outgroup, adjs in scenarios]

    with open('../output/gibbs_uniform_speaker100000_used.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    code_scenarios_speaker = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios_speaker = [adjs for ingroup, outgroup, adjs in scenarios]

    mturk_listener_scenarios, listener_qual_ratings = data_utils.merge_listener_with_ratings(
        code_scenarios_listener,
            ('../data/listener_filter_clean.csv', 15), ('../data/listener_semantic.csv', 8),  word_list,
        '../../graphics_gen/high_conf_listener.npy', counts=counts, scen_len=len(code_scenarios_listener))

    with open('../output/gibbs_uniform_listener100000_conf.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    code_scenarios_listener = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios_listener = [adjs for ingroup, outgroup, adjs in scenarios]

    mturk_speaker_scenarios, speaker_qual_ratings = data_utils.merge_speaker_with_ratings(adj_scenarios_speaker,
        ('../data/speaker_filter_clean.csv', 6), ('../data/speaker_semantic.csv', 13), '../../graphics_gen/high_conf_speaker.npy',
                                                                                    scen_len=len(scenarios), counts=counts)
    with open('../output/../output/gibbs_uniform_speaker100000_conf.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    code_scenarios_speaker = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios_speaker = [adjs for ingroup, outgroup, adjs in scenarios]

    return (code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, listener_qual_ratings), \
           (code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, speaker_qual_ratings)

def load_full_set_scen(single_adj=True):
    '''
    helper function to load semantic 5 noun 8 adj experiment data
    :param single_adj: only consider first adjective
    :return: (codename listener scenarios, adjective listener scenarios, mturk listener data, mturk listener confidence),
            (codename speaker scenarios, adjective speaker scenarios, mturk speaker data, mturk speaker confidence)
    '''
    data_set_str = 'combined'

    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'

    if data_set_str == 'glove':
        adjective_file = 'adj_glove.pickle'
    else:
        adjective_file = 'adj_choosen.pickle'

    word_list, adj_list, adj_code_prob_dict, scenarios, scenario_adjs = \
        data_utils.load_pickle_files(file_path, codenames_file, adjective_file)

    if data_set_str == 'combined':
        with open(file_path + 'adj_glove.pickle', 'rb') as f:
            glove_scenarios = pickle.load(f, encoding='latin-1')

        decoded_adj = []
        for k in range(len(glove_scenarios)):
            decoded_adj.append([adj.decode('utf-8') for adj in glove_scenarios[k]])
        glove_scenarios = decoded_adj

        with open(file_path + 'adj_choosen.pickle', 'rb') as f:
            bigram_scenarios = pickle.load(f, encoding='latin-1')

        decoded_adj = []
        for k in range(len(bigram_scenarios)):
            decoded_adj.append([adj.decode('utf-8') for adj in bigram_scenarios[k]])
        bigram_scenarios = decoded_adj

        scenario_adjs = bigram_scenarios + glove_scenarios

        scenarios = scenarios[:40] + scenarios[:40]

        assert (len(scenarios) == 80)
        assert (len(scenario_adjs) == 80)

    speaker_file = '../data/speaker_' + data_set_str + '.csv'

    listener_file = '../data/listener_' + data_set_str + '.csv'

    if data_set_str == 'combined':
        speaker_file = ['../data/speaker_bigram.csv', '../data/speaker_glove.csv']
        listener_file = ['../data/listener_bigram.csv', '../data/listener_glove.csv']

    listener_mturk_scenarios_bigram = data_utils.parse_listener(scenarios[:40], [listener_file[0]], scenario_adjs,
                                                                word_list, single_adj=False)
    listener_mturk_scenarios_glove = data_utils.parse_listener(scenarios[40:], [listener_file[1]], scenario_adjs,
                                                               word_list, single_adj=False)
    listener_mturk_scenarios = np.concatenate((listener_mturk_scenarios_bigram, listener_mturk_scenarios_glove), axis=0)
    speaker_mturk_scenarios, _ = data_utils.parse_speaker(speaker_file)

    if single_adj:
        listener_mturk_scenarios = listener_mturk_scenarios[:, 0:1, :]
        listener_mturk_scenarios = [scen / (scen.sum(axis=1).reshape(-1, 1)) for scen in listener_mturk_scenarios]
        return (scenarios, scenario_adjs, listener_mturk_scenarios),\
               (scenarios, scenario_adjs, speaker_mturk_scenarios)
    else:
        listener_mturk_scenarios = [scen / (scen.sum(axis=1).reshape(-1, 1)) for scen in listener_mturk_scenarios]
        code_scenario_listener = []
        adj_scenario_listener = []
        new_mturk_listener = []
        for i, scen in enumerate(scenarios):
            for j in range(8):
                new_mturk_listener.append(listener_mturk_scenarios[i][j])
                code_scenario_listener.append(scen)
                adj_scen = scenario_adjs[i]
                new_adj_scen = adj_scen[j:] + adj_scen[:j]
                adj_scenario_listener.append(new_adj_scen)
        new_mturk_listener = np.asarray(new_mturk_listener)
        new_mturk_listener = new_mturk_listener.reshape(len(listener_mturk_scenarios)*8, 1, -1)
        return (code_scenario_listener, adj_scenario_listener, new_mturk_listener), \
               (scenarios, scenario_adjs, speaker_mturk_scenarios)


def main(run_rsa=False, run_corr=False, gen_stats=False, experiment_num=3):
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'

    csv_keys = ['SIM TYPE', 'L_Literal top 1', 'ERR', 'L_Pragmatic top 1', 'ERR', 'L_Literal top 3', 'ERR',
                'L_Pragmatic top 3', 'ERR', 'LS_Literal top 1', 'ERR', 'LS_Literal top 3', 'ERR',
                'S_Literal top 1', 'ERR', 'S_Pragmatic top 1', 'ERR', 'S_Literal top 3', 'ERR',
                'S_Pragmatic top 3', 'ERR', 'SL_Literal top 1', 'ERR', 'SL_Literal top 3', 'ERR']

    word_list, adj_list, _, _, _ = \
        data_utils.load_pickle_files(file_path, codenames_file, 'adj_choosen.pickle')

    feature_str = 'google_bigram_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        bigram_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'twitter_bigram_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        twitter_bigram_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'conceptnet_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        conceptnet_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'wiki_glove_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        wiki_glove = pickle.load(fp, encoding='latin-1')

    feature_str = 'google_w2v_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        w2v_feats = pickle.load(fp, encoding='latin-1')

    feature_str = 'lda_norm_features.pickle'
    with open('../output/associative_features/' + feature_str, 'rb') as fp:
        lda_feats = pickle.load(fp, encoding='latin-1')

    if run_rsa:
        if experiment_num == 4:
            run_rsa_analysis_pragmatic([bigram_feats], ['bigram'], csv_keys, word_list)
        else:
            model_comp_list = [bigram_feats, conceptnet_feats, w2v_feats]
            model_comp_list_str = ['bigram_ls', 'conceptnet_ls', 'word2vec_ls']
            run_rsa_analysis(model_comp_list, model_comp_list_str, csv_keys, word_list, ex=experiment_num)

    elif run_corr:
        if experiment_num == 4:
            model_comp_list = [(bigram_feats, 'ls'), (bigram_feats, 'ps')]
            run_model_probability_prag(model_comp_list, rank_corr=True)
        else:
            model_comp_list = [(bigram_feats, 'ls'), (conceptnet_feats, 'ls'), (w2v_feats, 'ls')]
            run_model_probability(model_comp_list, rank_corr=True, ex=experiment_num)
    elif gen_stats:
        if experiment_num == 4:
            model_comp_list = [(bigram_feats, 'ls'), (bigram_feats, 'ps')]
            model_comp_list_str = ['Bigram_lit', 'Bigram_prag']
            gen_speaker_stats(model_comp_list, model_comp_list_str, exp=experiment_num, metric='corr')
            model_comp_list = [(bigram_feats, 'll'), (bigram_feats, 'pl')]
            gen_listener_stats(model_comp_list, model_comp_list_str, exp=experiment_num, metric='corr')

        else:
            model_comp_list = [(bigram_feats, 'll'), (conceptnet_feats, 'll'), (w2v_feats, 'll')]
            model_comp_list_str = ['Bigram', 'Conceptnet', 'Word2Vec']
            gen_speaker_stats(model_comp_list, model_comp_list_str, exp=experiment_num, metric='corr')

            model_comp_list = [(bigram_feats, 'ls'), (conceptnet_feats, 'ls'), (w2v_feats, 'ls')]
            gen_listener_stats(model_comp_list, model_comp_list_str, exp=experiment_num, metric='corr')


if __name__ == '__main__':
    main(run_corr=True, experiment_num=4)