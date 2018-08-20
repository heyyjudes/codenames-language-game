# Tools for calculating success of gameplay between
# speakers and listeners (both RSA model and MTurk)

import data_utils
import json
import csv
DEBUG = True

def find_all_vals_dict(target_val, input_dict):
    '''
    find all indices of given value when given dictionary
    :param target_val: value to match
    :param input_dict: input dictionary containing indices and values
    :return: array of indices where the list matches given value
    '''
    val_arr = []
    for key in input_dict.keys():
        if input_dict[key] == target_val:
            val_arr.append(key)
    return val_arr


def find_all_vals_list(target_val, input_list):
    '''
    find all indices of given value when given list
    :param target_val: value to match
    :param input_list: input list containing values
    :return: array of indices where the list matches given value
    '''
    val_arr = []
    for i in range(len(input_list)):
        if input_list[i] == target_val:
            val_arr.append(i)
    return val_arr


def calc_match_score(input_code_scenarios, input_speaker_dict, input_listener_dict, input_entropy_dict):
    '''
    compute accuracy score of gameplay given speaker and listener dictionary
    :param input_code_scenarios: list of noun scenarios
    :param input_speaker_dict: dictionary containing speaker probabilities for each adjective scenario
    :param input_listener_dict: dictionary containing dictionary of listener probabilities for each noun pair scenario
    :param input_entropy_dict: dictionary containing boolean values of whether entropy is bounded
    :return:
    '''

    # initialize counts
    single_clue_single_answer_match = 0
    bounded_single_clue_single_answer_match = 0
    single_clue_multiple_answer_match = 0
    multiple_clue_single_answer_match = 0
    multiple_clue_multiple_answer_match = 0
    single_clue_wrong = 0
    multiple_clue_wrong = 0
    total_bounded_entropy_speakers = 0
    results_arr = []

    for key in input_speaker_dict.keys():
        # check if entropy is bounded
        if list(input_entropy_dict[key].values())[0]:
            bounded_speaker_entropy = True
            total_bounded_entropy_speakers += 1
        else:
            bounded_speaker_entropy = False

        match = False
        [[n_0, n_1], [_, _, _]] = input_code_scenarios[int(key)]
        in_words = str((min(n_0, n_1), max(n_0, n_1)))
        max_list = find_all_vals_dict(max(input_speaker_dict[key].values()), input_speaker_dict[key])
        if len(max_list) == 1:
            # speaker produces only single clue
            pair_prob = input_listener_dict[key][max_list[0]]
            max_comb_list = find_all_vals_dict(max(pair_prob.values()), pair_prob)
            if len(max_comb_list) == 1 and max_comb_list[0] == in_words:
                # listener has only single max prob answer
                    single_clue_single_answer_match += 1
                    match = True
                    if bounded_speaker_entropy:
                        bounded_single_clue_single_answer_match += 1
            elif len(max_comb_list) > 1 and in_words in max_comb_list:
                # listener has multiple equally likely answers
                    single_clue_multiple_answer_match += 1
            else:
                #
                single_clue_wrong += 1
        else:
            # speaker produces multiply clues with identical probability s: usually only happens with AMAX and Mturk
            max_comb_list = []
            for j in range(len(max_list)):
                pair_prob = input_listener_dict[key][max_list[j]]
                max_comb_list += find_all_vals_dict(max(pair_prob.values()), pair_prob)
            # only 1 max value
            if len(max_comb_list) == 1 and max_comb_list[0] == in_words:
                    # listener has only single max prob answer
                    multiple_clue_single_answer_match += 1
            elif len(max_comb_list) > 1 and in_words in max_comb_list:
                    # listener has multiple equally likely answers
                    multiple_clue_multiple_answer_match += 1
            else:
                multiple_clue_wrong += 1

        results_arr.append(int(match))
    total_len = len(input_code_scenarios)

    if DEBUG:
        # show stats for all categories of matches
        print('single clue match')
        print(single_clue_single_answer_match)
        print(single_clue_single_answer_match/total_len)

        print('bounded single clue match')
        print(bounded_single_clue_single_answer_match)
        print(bounded_single_clue_single_answer_match/float(total_bounded_entropy_speakers))

        print('single clue multiple answer match')
        print(single_clue_multiple_answer_match)
        print(single_clue_multiple_answer_match/total_len)

        print('multiple clue answer match')
        print(multiple_clue_single_answer_match)
        print(multiple_clue_single_answer_match/total_len)
        print(multiple_clue_multiple_answer_match)
        print(multiple_clue_multiple_answer_match/total_len)

        print('single clue mismatch')
        print(single_clue_wrong)
        print(single_clue_wrong/total_len)

        print('multiple clue mismatch')
        print(multiple_clue_wrong)
        print(multiple_clue_wrong/total_len)

        print("not considered")
        not_considered = single_clue_multiple_answer_match + multiple_clue_single_answer_match + multiple_clue_multiple_answer_match
        print("not considered", not_considered)

    total_correct = single_clue_single_answer_match
    total_wrong = single_clue_wrong + multiple_clue_wrong
    return total_correct/total_len, total_wrong, results_arr


if __name__ == "__main__":

    results_keys = ['ml_ms', 'ml_ls', 'ml_ps', 'll_ms', 'll_ls', 'll_ps', 'pl_ms', 'pl_ls', 'pl_ps']

    # comp_str = 'combined' for both sets of answers, 'glove' to only consider glove set,
    # 'bigram' to only consider glove set

    comp_str = 'combined'
    if comp_str == 'glove':
        adjective_file = 'adj_glove.pickle'

    else:
        adjective_file = 'adj_choosen.pickle'

    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'

    for likelihood_str in ['bigram', 'glove', 'concept']:
        csv_file = open('../results/gameplay' + likelihood_str + '.csv', 'a')
        results_writer = csv.writer(csv_file)
        results_writer.writerow(results_keys)

        m = 'PROD'
        val = '1.0'
        with open('../output/prob_dictionary/listener_' + likelihood_str + '_'+ val + m + '.json', 'rb') as fp:
            listener_dict = json.load(fp, encoding='latin-1')

        with open('../output/prob_dictionary/speaker_' + likelihood_str + '_' + val + m + '.json', 'rb') as fp:
            speaker_dict = json.load(fp, encoding='latin-1')

        codes_list, adj_list, adj_code_prob_dict, code_scenarios, adj_scenarios = \
            data_utils.load_pickle_files(file_path, codenames_file, adjective_file)

        if comp_str == 'combined':
            code_scenarios = code_scenarios[:40] + code_scenarios[:40]


        print('01: mturk listener and mturk speaker')
        v_1, _, result_arr_1 = calc_match_score(code_scenarios, speaker_dict['mturk'], listener_dict['mturk'], speaker_dict['entropy'])

        print('02: mturk listener and literal speaker')
        v_2, _, result_arr_2 = calc_match_score(code_scenarios, speaker_dict['ls'], listener_dict['mturk'], speaker_dict['entropy'])

        print('03: mturk listener and pragmatic speaker')
        v_3, _, result_arr_3 = calc_match_score(code_scenarios, speaker_dict['ps'], listener_dict['mturk'], speaker_dict['entropy'])

        print('04: literal listener and mturk speaker')
        v_4, _, result_arr_4 = calc_match_score(code_scenarios, speaker_dict['mturk'], listener_dict['ll'], speaker_dict['entropy'])

        print('05: literal listener and listener speaker')
        v_5, _, result_arr_5 = calc_match_score(code_scenarios, speaker_dict['ls'], listener_dict['ll'], speaker_dict['entropy'])

        print('06: literal listener and pragmatic speaker')
        v_6, _, result_arr_6 = calc_match_score(code_scenarios, speaker_dict['ps'], listener_dict['ll'], speaker_dict['entropy'])

        print('07: pragmatic listener and mturk speaker')
        v_7, _, result_arr_7 = calc_match_score(code_scenarios, speaker_dict['mturk'], listener_dict['pl'], speaker_dict['entropy'])

        print('08: pragmatic listener and literal speaker')
        v_8, _, result_arr_8 = calc_match_score(code_scenarios, speaker_dict['ls'], listener_dict['pl'], speaker_dict['entropy'])

        print('09: pragmatic listener and pragmatic speaker')
        v_9, _, result_arr_9 =calc_match_score(code_scenarios, speaker_dict['ps'], listener_dict['pl'], speaker_dict['entropy'])

        results_writer.writerow([v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9])
        csv_file.close()

        csv_file = open('../results/gameplay_by_scenario_' + likelihood_str + '.csv', 'a')
        results_writer = csv.writer(csv_file)
        results_writer.writerow(results_keys)

        for i in range(80):
            new_row = [result_arr_1[i], result_arr_2[i], result_arr_3[i], result_arr_4[i], result_arr_5[i],
                       result_arr_6[i], result_arr_7[i], result_arr_8[i], result_arr_9[i]]
            results_writer.writerow(new_row)
        csv_file.close()





