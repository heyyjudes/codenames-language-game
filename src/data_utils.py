# helper methods for parsing experimental results
import pickle
import itertools
import numpy as np
import csv
import pandas as pd
import math

NUM_ADJ= 8
NUM_NOUN = 5
NUM_NPAIR = 10

def load_pickle_files(path, freq_filename, adj_filename):
    '''
    :param path: path to pickle files
    :param freq_filename: filename of code scenarios and bigram probabilities
    :param adj_filename: filename of adjective scenario pickle
    :return: list of codes, list of adjectives, code-adj probabilities, code_scenarios,
    adjective scenarios
    '''
    with open(path + freq_filename, 'rb') as f:
        all_list = pickle.load(f, encoding='latin-1')
    adj_list = all_list['adjs']
    codes_list = all_list['codes']
    # each entry is len 50 vec for 50 nouns
    adj_code_prob_dict = all_list['adj_feats']

    # List code scenarios
    # each list has [[n_0, n_1], [n_2, n_3, n_4]]
    code_scenarios = all_list['scenarios']
    # list of adjective scenarios
    with open(path + adj_filename, 'rb') as f:
        adj_scenarios = pickle.load(f, encoding='latin-1')

    decoded_adj = []
    for k in range(len(adj_scenarios)):
        decoded_adj.append([adj.decode('utf-8') for adj in adj_scenarios[k]])
    adj_scenarios = decoded_adj

    return codes_list, adj_list, adj_code_prob_dict, code_scenarios, adj_scenarios


def record_values(val_mat, starting_index, row, num_adj, scenario_combinations, adj_scenarios, word_list,
                  scenario_dict):
    '''
    count values by QID
    :param val_mat: empty matrix containing
    :param starting_index: starting question index
    :param row: row of csv file
    :param num_adj: number of adjectives
    :param scenario_combinations: list of scenario combinations
    :param adj_scenarios: list of adjective scenarios
    :param word_list: list of codes
    :return:
    '''
    #adj_index = int(starting_index/9 - 1)
    adj_index = int((starting_index - 9)/9)
    scenario = scenario_combinations[adj_index]
    adj_scenario = adj_scenarios[adj_index]
    for i in range(num_adj):
        q_str = "QID" + str(i + starting_index)
        noun_0, noun_1 = row[q_str].split(',')
        ind_0 = word_list.index(noun_0)
        ind_1 = word_list.index(noun_1)
        if ind_0 > ind_1:
            noun_key = (ind_1, ind_0)
        else:
            noun_key = (ind_0, ind_1)
        val_mat[adj_index, i, scenario.index(noun_key)] += 1
        curr_adj = adj_scenario[i]
        scenario_dict[adj_index][curr_adj][noun_key] += 1


def parse_listener(code_scenarios, inverse_file_arr, adj_scenarios, codes_list, with_pair_labels=False, single_adj=False):
    '''
    parses listener experiment results file
    :param code_scenarios: list of codename scenarios
    :param inverse_file: file containing experiment results
    :param adj_scenarios: list of adjective scenarios
    :param codes_list: list of codenames
    :return: matrix of answer counts
    '''
    # Parse Inverse Game CSV
    num_scenario = len(code_scenarios)

    beg_index = 9

    if single_adj:
        num_adj = 1
        end_index =  beg_index + 2 * num_scenario
        initial_index_list = range(beg_index, end_index, 2)
    else:
        num_adj = len(adj_scenarios[0])
        end_index = (num_adj + 1) * (num_scenario + 1)
        initial_index_list = range(beg_index, end_index-1, num_adj+1)

    # create list of combinations
    scenario_dict = {}
    scenario_comb = []
    for s in range(len(code_scenarios)):
        new_samples = []
        out_group = [code_scenarios[s][1][i] for i in range(len(code_scenarios[s][1]))]
        in_group = code_scenarios[s][0]

        s_list = in_group + out_group

        # generate all possible combinations of codewords
        samples = list(itertools.combinations(s_list, 2))
        for (ind_0, ind_1) in samples:
            if ind_1 < ind_0:
                # order combination indices
                temp = ind_0
                ind_0 = ind_1
                ind_1 = temp
            new_samples.append((ind_0, ind_1))
        adjective_dict = {}
        for adj in adj_scenarios[s]:
            new_dict = dict((el, 0) for el in new_samples)
            adjective_dict[adj] = new_dict
        scenario_dict[s] = adjective_dict
        scenario_comb.append(new_samples)

    val_matrix = np.zeros((num_scenario, num_adj, len(samples)))

    for inverse_file in inverse_file_arr:
        with open(inverse_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                for ind in initial_index_list:
                    check_str = "QID" + str(ind)
                    if len(row[check_str]) > 0:
                        record_values(val_matrix, ind, row, num_adj, scenario_comb, adj_scenarios, codes_list, scenario_dict)
        if with_pair_labels:
            return val_matrix, scenario_dict
        else:
            return val_matrix


def merge_listener_with_ratings(code_scenarios, total_listener_file, filtered_listener_file, codes_list, conf_arr_file,
                                counts=False, num_perm=1, scen_len=120, match_arr_file=None):
    # results
    total_listener_file, beg_id, = total_listener_file
    df = pd.read_csv(total_listener_file)

    total_scen = len(code_scenarios)

    result_beg = beg_id
    result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    result_df = df[result_arr]

    qual_beg = result_beg + 1
    qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    qual_df = df[qual_arr]

    conf_arr = np.load(conf_arr_file)
    filtered_listener_file, beg_id, = filtered_listener_file
    filtered_df = pd.read_csv(filtered_listener_file)

    total_scen = int((len(filtered_df.keys())-2)/2)
    result_beg = beg_id
    filtered_result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    filtered_result_df = filtered_df[filtered_result_arr]

    qual_beg = result_beg + 1
    filtered_qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    filtered_qual_df = filtered_df[filtered_qual_arr]

    filtered_ind = 0
    filtered_code_scenarios = []
    for i, id_str in enumerate(result_arr):
        if i%scen_len not in conf_arr:
            result_df = result_df.drop(id_str, axis=1)
            qual_df = qual_df.drop(qual_arr[i], axis=1)
        else:
            result_df = result_df.rename(index=str, columns={id_str: filtered_result_arr[filtered_ind]})
            qual_df = qual_df.rename(index=str, columns={qual_arr[i]: filtered_qual_arr[filtered_ind]})
            filtered_code_scenarios.append(code_scenarios[i])
            filtered_ind += 1

    merged_df = pd.concat([result_df, filtered_result_df], axis=0)
    qual_merged_df = pd.concat([qual_df, filtered_qual_df], axis=0)
    average_qual = list(qual_merged_df.mean())
    mturk_scen = []
    scenario_dict = []
    adj_counts_arr = []
    for i, id_str in enumerate(filtered_result_df.keys()):
        
        in_group, out_group = filtered_code_scenarios[i]
        s_list = in_group + out_group
        # generate all possible combinations of codewords
        samples = list(itertools.combinations(s_list, 2))
        agg_counts = merged_df[id_str].value_counts()
        #print(id_str, agg_counts)
        npair_counts = []
        npair_labels = []
        for ind_0, ind_1 in samples:
            loc_str_1 = codes_list[min(ind_0, ind_1)] + ',' + codes_list[max(ind_0, ind_1)]
            loc_str_2 = codes_list[max(ind_0, ind_1)] + ',' + codes_list[min(ind_0, ind_1)]
            if loc_str_1 in agg_counts.index:
                npair_counts.append(agg_counts.loc[loc_str_1])
                npair_labels.append(loc_str_1)
            elif loc_str_2 in agg_counts.index:
                npair_counts.append(agg_counts.loc[loc_str_2])
                npair_labels.append(loc_str_2)
            else:
                npair_counts.append(0)
        if not counts:
            normalized_counts = np.asarray(npair_counts) / sum(npair_counts)
            mturk_scen.append(normalized_counts)
            scenario_map = dict(zip(npair_labels, normalized_counts))
            
        else:
            mturk_scen.append(npair_counts)
            scenario_map = dict(zip(npair_labels, npair_counts))
        
        #print(scenario_map)
        scenario_dict.append(scenario_map)
        adj_counts_arr.append(sum(npair_counts))

    mturk_arr = np.asarray(mturk_scen).reshape(total_scen, 1, len(samples))
    filtered_mturk_arr = []
    filtered_average_qual = []
    if match_arr_file:
        match_arr = np.load(match_arr_file)
        for i in range(len(mturk_arr)):
            if i%(total_scen/num_perm) in match_arr:
                filtered_mturk_arr.append(mturk_arr[i])
                filtered_average_qual.append(average_qual[i])
        mturk_arr = np.asarray(filtered_mturk_arr)
        average_qual = np.asarray(average_qual)

    return mturk_arr, average_qual, scenario_dict

def parse_listener_with_ratings(code_scenarios, inverse_file, codes_list, counts=False):
    '''
    parses listener experiment results file
    :param code_scenarios: list of codename scenarios
    :param inverse_file: file containing experiment results
    :param adj_scenarios: list of adjective scenarios
    :param codes_list: list of codenames
    :return: matrix of answer counts
    '''
    # Parse Inverse Game CSV
    inverse_file, result_beg = inverse_file
    df = pd.read_csv(inverse_file)
    total_scen = len(code_scenarios)
    result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    result_df = df[result_arr]

    qual_beg = result_beg + 1
    qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    qual_df = df[qual_arr]
    average_qual = list(qual_df.mean())
    mturk_scen = []
    for i, id_str in enumerate(result_arr):
        in_group, out_group = code_scenarios[i]
        s_list = in_group + out_group

        # generate all possible combinations of codewords
        samples = list(itertools.combinations(s_list, 2))
        agg_counts = result_df[id_str].value_counts()
        npair_counts = []
        for ind_0, ind_1 in samples:
            loc_str_1 = codes_list[min(ind_0, ind_1)] + ',' + codes_list[max(ind_0, ind_1)]
            loc_str_2 = codes_list[max(ind_0, ind_1)] + ',' + codes_list[min(ind_0, ind_1)]
            if loc_str_1 in agg_counts.index:
                npair_counts.append(agg_counts.loc[loc_str_1])
            elif loc_str_2 in agg_counts.index:
                npair_counts.append(agg_counts.loc[loc_str_2])
            else:
                npair_counts.append(0)
        if not counts:
            normalized_counts = np.asarray(npair_counts) / sum(npair_counts)
            mturk_scen.append(normalized_counts)
        else:
            mturk_scen.append(npair_counts)

    return np.expand_dims(mturk_scen, axis=1), average_qual


def parse_speaker(speaker_file_arr):
    '''
    parse speaker file
    :param speaker_file: file containing speaker data
    :return: scenario fraction and answer count for each scenario
    '''
    answer_counts = []
    line_count = 0
    scenarios_mturks = []
    for speaker_file in speaker_file_arr:
        with open(speaker_file, 'r') as f:
            for line in f:
                if line.startswith('mTurkCode'):
                    break 
                else: 
                    line = line.rstrip("\n")
                    fields = line.split(',')
                    if len(fields) == 4: 
                        if len(fields[3]) > 0 and fields[0] not in ['#']:
                            if len(fields[0]) > 0:
                                q = int(fields[0])
                                if q == 1:
                                    line_count = 0
                                    scen = []
                                scen.append(int(fields[3]))
                                line_count += int(fields[3])
                                if q == NUM_ADJ:
                                    answer_counts.append(line_count)
                                    scenarios_mturks.append(np.array(scen, dtype=float))

        scenarios_mturks = [mscen / mscen.sum(axis=0, keepdims=True) for mscen in scenarios_mturks]
    return scenarios_mturks, answer_counts

def merge_speaker_with_ratings(adj_scenarios, total_speaker_file, filtered_speaker_file, conf_arr_file, counts=False,
                               num_perm=1, scen_len=120, match_arr_file=None):
    total_speaker_file, beg_id, = total_speaker_file
    df = pd.read_csv(total_speaker_file)
    total_scen = len(adj_scenarios)
    result_beg = beg_id
    result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    result_df = df[result_arr]

    qual_beg = result_beg + 1
    qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    qual_df = df[qual_arr]

    conf_arr = np.load(conf_arr_file)
    filtered_speaker_file, beg_id, = filtered_speaker_file
    filtered_df = pd.read_csv(filtered_speaker_file)

    total_scen = int((len(filtered_df.keys())-2)/2)
    result_beg = beg_id
    filtered_result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    filtered_results_df = filtered_df[filtered_result_arr]

    qual_beg = result_beg + 1
    filtered_qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    filtered_qual_df = filtered_df[filtered_qual_arr]

    filtered_ind = 0
    filtered_adj_scenarios = []
    for i, id_str in enumerate(result_arr):
        if i%scen_len not in conf_arr:
            result_df = result_df.drop(id_str, axis=1)
            qual_df = qual_df.drop(qual_arr[i], axis=1)
        else:
            result_df = result_df.rename(index=str, columns={id_str: filtered_result_arr[filtered_ind]})
            qual_df = qual_df.rename(index=str, columns={qual_arr[i]: filtered_qual_arr[filtered_ind]})
            filtered_adj_scenarios.append(adj_scenarios[i])
            filtered_ind += 1

    merged_df = pd.concat([result_df, filtered_results_df], axis=0)
    qual_merged_df = pd.concat([qual_df, filtered_qual_df], axis=0)
    average_qual = list(qual_merged_df.mean())

    mturk_scen = []
    adj_counts_arr = []
    for i, id_str in enumerate(filtered_result_arr):
        adjs = filtered_adj_scenarios[i]
        agg_counts = merged_df[id_str].value_counts()
        adj_counts = []
        for a in adjs:
            try:
                adj_counts.append(agg_counts.loc[a])
            except:
                adj_counts.append(0)
        if not counts:
            normalized_counts = np.asarray(adj_counts / sum(adj_counts))
            mturk_scen.append(normalized_counts)
        else:
            mturk_scen.append(adj_counts)
        adj_counts_arr.append(sum(adj_counts))

    # print("max:", max(adj_counts_arr))
    # print("min", min(adj_counts_arr))
    # print("mean: ", np.mean(adj_counts_arr))
    mturk_arr = mturk_scen
    filtered_mturk_arr = []
    filtered_average_qual = []

    if match_arr_file:
        match_arr = np.load(match_arr_file)
        for i in range(len(mturk_arr)):
            if i%(total_scen/num_perm) in match_arr:
                filtered_mturk_arr.append(mturk_arr[i])
                filtered_average_qual.append(average_qual[i])
        mturk_scen = np.asarray(filtered_mturk_arr)
        average_qual = np.asarray(average_qual)

    return mturk_scen, average_qual

def parse_speaker_with_ratings(speaker_file, scenario_adjs, counts=False):
    speaker_file, result_beg = speaker_file
    df = pd.read_csv(speaker_file)
    total_scen = len(scenario_adjs)

    result_arr = ['QID' + str(i) for i in range(result_beg, 3 * total_scen + result_beg, 3)]
    result_df = df[result_arr]
    mturk_scen = []

    qual_beg = result_beg + 1
    qual_arr = ['QID' + str(i) for i in range(qual_beg, 3 * total_scen + qual_beg, 3)]
    qual_df = df[qual_arr]
    average_qual = list(qual_df.mean())

    for i, id_str in enumerate(result_arr):
        adjs = scenario_adjs[i]
        agg_counts = result_df[id_str].value_counts()
        adj_counts = []
        for a in adjs:
            try:
                adj_counts.append(agg_counts.loc[a])
            except:
                adj_counts.append(0)
        if not counts:
            normalized_counts = np.asarray(adj_counts / sum(adj_counts))
            mturk_scen.append(normalized_counts)
        else:
            mturk_scen.append(adj_counts)
    return mturk_scen, average_qual

def combine_experiments(file_path, scenarios):
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

    return scenarios, scenario_adjs


