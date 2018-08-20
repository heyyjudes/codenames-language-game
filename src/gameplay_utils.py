import numpy as np
import itertools


def top_match_gameplay(listener_data, speaker_data):
    if len(listener_data) == 4:
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data
    else:
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios = speaker_data
    correct_count = 0
    speaker_tie = 0
    listener_tie = 0
    for i, scen in enumerate(mturk_speaker_scenarios):
        print(i)
        adjs = adj_scenarios_speaker[i]
        # take the current listener codes
        if len(adjs) == 8:
            codes = code_scenarios_listener[8*i]
        else:
            codes = code_scenarios_listener[i]
        answer_adj = adjs[np.argmax(scen)]

        if len(np.where(scen == max(scen))[0]) > 1:
            speaker_tie += 1
        else:
            for j, code_scen in enumerate(code_scenarios_listener):
                if codes == code_scen and answer_adj == adj_scenarios_listener[j][0] \
                    and set(adj_scenarios_listener[j]) == set(adjs):
                    if len(np.where(mturk_listener_scenarios[j][0] == max(mturk_listener_scenarios[j][0]))[0]) > 1:
                        listener_tie += 1
                    #print("listener", code_scenarios_listener[j], adj_scenarios_listener[j])
                    #print("speaker", code_scenarios_speaker[i], adj_scenarios_speaker[i])
                    out_group = [code_scen[1][k] for k in range(len(code_scen[1]))]
                    in_group = [code_scen[0][0], code_scen[0][1]]
                    sample_list = in_group + out_group
                    noun_pairs = list(itertools.combinations(sample_list, 2))
                    a_0, a_1 = noun_pairs[np.argmax(mturk_listener_scenarios[j][0])]
                    if a_0 in code_scenarios_speaker[i][0] and a_1 in code_scenarios_speaker[i][0]:
                        correct_count += 1
                        print("match")
    print(correct_count, speaker_tie, listener_tie)


def prob_gameplay(listener_data, speaker_data):
    if len(listener_data) == 4:
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios, _ = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios, _ = speaker_data
    else:
        code_scenarios_listener, adj_scenarios_listener, mturk_listener_scenarios = listener_data
        code_scenarios_speaker, adj_scenarios_speaker, mturk_speaker_scenarios = speaker_data
    total_prob = 0
    for i, scen in enumerate(mturk_speaker_scenarios):
        print(i)
        adjs = adj_scenarios_speaker[i]
        if len(adjs) == 8:
            codes = code_scenarios_listener[8*i]
        else:
            codes = code_scenarios_listener[i]
        adj_prob = 0
        for a, adj in enumerate(adjs):
            prob = 0
            for j, code_scen in enumerate(code_scenarios_listener):

                if codes == code_scen and adj == adj_scenarios_listener[j][0] and set(adj_scenarios_listener[j]) == set(adjs):

                    out_group = [code_scen[1][k] for k in range(len(code_scen[1]))]
                    in_group = [code_scen[0][0], code_scen[0][1]]
                    sample_list = in_group + out_group

                    noun_pairs = list(itertools.combinations(sample_list, 2))
                    a_0, a_1 = code_scenarios_speaker[i][0]
                    if (a_0, a_1) in noun_pairs:
                        ind = noun_pairs.index((a_0, a_1))
                        prob = mturk_listener_scenarios[j][0][ind]*mturk_speaker_scenarios[i][a]
                    elif (a_1, a_0) in noun_pairs:
                        ind = noun_pairs.index((a_1, a_0))
                        prob = mturk_listener_scenarios[j][0][ind]*mturk_speaker_scenarios[i][a]
                    else:
                        print(code_scen)
                        print(codes)
                        print(a_0, a_1, "not found")
                        print(noun_pairs)
            #print(prob)
            #print(mturk_listener_scenarios[j])
            adj_prob += prob
        print(adj_prob)
        total_prob += adj_prob
    print(total_prob/len(mturk_speaker_scenarios))
