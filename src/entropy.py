# Tools for entropy calculations of speaker and listener responses
import numpy as np
import scipy.stats
import data_utils
import rsa_model
import matplotlib.pyplot as plt
import pickle


NUM_ADJ= 4
NUM_NOUN = 3
NUM_NPAIR = 3

def random_sample_entropy(response_arr, response_dist, num_responses, sample_count=10000):
    '''
    calcuate entropy of a given response
    :param response_arr: array of responses
    :param response_dist: distribution of responses
    :param num_responses: number of possible responses
    :param sample_count: number of samples to take
    :return: entropy of each sample for sample count
    '''
    total_ent = []
    for i in range(sample_count):
        sample_arr = np.random.multinomial(1, response_dist)
        sample_length = np.argwhere(sample_arr)
        sample_length = response_arr[np.sum(sample_length)]
        arr = np.random.multinomial(sample_length, [1.0 / num_responses] * num_responses)
        ent = scipy.stats.entropy(arr)
        total_ent.append(ent)
    return np.asarray(total_ent)


def get_response_distribution(val_matrix, num_scenarios=40, scenario_arr=None):
    '''
    get number of responses and distributions
    :param val_matrix: count matrix of results
    :param num_scenarios: number of total scenarios
    :param scenario_arr: optional list of scenario answer count
    :return: array of number of responses, array of frequency of response number
    '''
    sample_dist = {}
    for i in range(num_scenarios):
        if scenario_arr:
            curr_sum = scenario_arr[i]
        else:
            curr_sum = np.sum(val_matrix[i, :, :], axis=1)[0]
        if curr_sum in sample_dist.keys():
            sample_dist[curr_sum] += 1
        else:
            sample_dist[curr_sum] = 1
    r_keys = sample_dist.keys()
    r_vals = []
    r_keys = [int(key) for key in r_keys]
    for key in r_keys:
        r_vals.append(sample_dist[key] / float(num_scenarios))
    return r_keys, r_vals


def get_data_entropy(val_matrix, num_dim=3):
    '''
    calculate entropy of data
    :param val_matrix: matrix of result count
    :param num_dim: optional number of dimensions of val_matrix
    :return: list of num_scenarioxnum_adj entropy
    '''
    entropy_list = []
    rand_entropy = np.zeros((np.asarray(val_matrix).shape[-1],))
    rand_entropy.fill(1/np.asarray(val_matrix).shape[-1])
    print("uniform entropy", scipy.stats.entropy(rand_entropy))
    if num_dim == 3:
        for i in range(len(val_matrix)):
            curr_mat_sum = np.sum(val_matrix[i, :, :], axis=1)[0]
            curr_mat = val_matrix[i, :, :] / curr_mat_sum
            for j in range(len(curr_mat)):
                entropy = scipy.stats.entropy(curr_mat[j])
                entropy_list.append(entropy)
    elif num_dim == 2:
        for i in range(len(val_matrix)):
            entropy = scipy.stats.entropy(val_matrix[i])
            entropy_list.append(entropy)

    return entropy_list


def create_histogram(entropy_list, save_str):
    '''
    create histogram of input entropy values
    :param entropy_list: list of entropy values to plot on histogram
    :param save_str: string to save file as
    :return: none
    '''
    data_mean = np.mean(entropy_list)
    plt.hist(entropy_list, bins='auto')  # arguments are passed to np.histogram
    plt.axvline(x=data_mean, color='k', label='Random Mean')
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.savefig(save_str)
    plt.clf()
    return


def create_cdf(entropy_list, save_str, r_keys, r_vals, num_resp):
    '''
    create cdf comparison of random vs data entropy
    :param entropy_list: list of data entropy
    :param save_str: string to save figure as
    :param r_keys: list of number of possible responses
    :param r_vals: distribution of number of possible responses
    :return:
    '''

    # method 1
    H, X1 = np.histogram(entropy_list, bins=50, normed=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H) * dx

    # method 2
    random_sample = random_sample_entropy(r_keys, r_vals, num_resp, 10000)
    H2, X2 = np.histogram(random_sample, bins=50, normed=True)
    dx = X2[1] - X2[0]
    F2 = np.cumsum(H2) * dx

    # find fifth percentile value
    sorted_random_sample = np.sort(random_sample)
    print("random mean ", np.mean(sorted_random_sample))

    fifth_val = sorted_random_sample[int(0.05 * len(sorted_random_sample))]
    print("num less than 5th percentile", len(np.where(entropy_list < fifth_val)[0]))
    print("num less than mean", len(np.where(entropy_list < np.mean(sorted_random_sample))[0]))
    print("random fifth percent", fifth_val)
    print("max ent: ", np.max(entropy_list))
    plt.plot(X1[1:], F1, label='Experimental')
    plt.plot(X2[1:], F2, label='Random', color='r')
    plt.axvline(x=fifth_val, linestyle='--', color='r', label='5th Percentile of Random')
    plt.legend()
    plt.xlabel("Entropy")
    plt.ylabel("Cumulative fraction")
    plt.savefig(save_str)
    plt.clf()
    return


if __name__ == "__main__":
    adjective_file = 'adj_choosen.pickle'
    codenames_file = 'noun_adjectives.pickle'
    file_path = '../output/'
    # listener_file = ['../data/listener_glove.csv']
    # speaker_file = ['../data/speaker_glove.csv']
    #
    # codes_list, adj_list, adj_code_prob_dict, code_scenarios, adj_scenarios = \
    #     data_utils.load_pickle_files(file_path, codenames_file, adjective_file)
    #
    #
    # n = 40
    # code_scenarios = code_scenarios[:n]
    # adj_scenarios = adj_scenarios[:n]
    #
    # listener_matrix = data_utils.parse_listener(code_scenarios, listener_file, adj_scenarios, codes_list)
    # listener_entropy = get_data_entropy(listener_matrix, num_dim=3)
    # response_keys, response_vals = get_response_distribution(listener_matrix)
    # mean = sum([response_vals[i]*response_keys[i] for i in range(len(response_vals))])
    # create_histogram(listener_entropy, '../img/listener_glove_histogram.pdf')
    # create_cdf(listener_entropy, '../img/listener_glove_cdf.pdf', response_keys, response_vals)
    # print("mean", np.mean(listener_entropy))
    # print("variance", np.var(listener_entropy))
    #
    # speaker_matrix, scenario_count = data_utils.parse_speaker(speaker_file)
    # speaker_entropy = get_data_entropy(speaker_matrix, num_dim=2)
    # response_keys, response_vals = get_response_distribution(listener_matrix, scenario_arr=scenario_count)
    # mean = sum([response_vals[i]*response_keys[i] for i in range(len(response_vals))])
    # create_histogram(speaker_entropy, '../img/speaker_glove_histogram.pdf')
    # create_cdf(speaker_entropy, '../img/speaker_glove_cdf.pdf', response_keys, response_vals)
    # print("mean", np.mean(speaker_entropy))
    # print("variance", np.var(speaker_entropy))

    listener_file = ['../data/listener_uniform.csv']
    speaker_file = ['../data/speaker_uniform.csv']

    codes_list, adj_list, _, _, _ = \
        data_utils.load_pickle_files(file_path, codenames_file, adjective_file)

    with open('../output/gibbs_uniform_speaker50000_used.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    n = 40
    code_scenarios = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios = [adjs for ingroup, outgroup, adjs in scenarios]

    speaker_matrix, scenario_count = data_utils.parse_speaker(speaker_file)
    speaker_entropy = get_data_entropy(speaker_matrix, num_dim=2)
    response_keys, response_vals = get_response_distribution(speaker_matrix, num_scenarios=len(code_scenarios), scenario_arr=scenario_count)
    mean = sum([response_vals[i]*response_keys[i] for i in range(len(response_vals))])
    create_histogram(speaker_entropy, '../img/speaker_uniform_histogram.pdf')
    create_cdf(speaker_entropy, '../img/speaker_uniform_cdf.pdf', response_keys, response_vals, num_resp=NUM_ADJ)
    print("mean", np.mean(speaker_entropy))
    print("variance", np.var(speaker_entropy))

    with open('../output/gibbs_uniform_listener50000_used.pickle', 'rb') as f:
        scenarios = pickle.load(f)
    scenarios = scenarios['scenarios']

    n = 40
    code_scenarios = [[ingroup, outgroup] for ingroup, outgroup, adjs in scenarios]
    adj_scenarios = [adjs for ingroup, outgroup, adjs in scenarios]

    listener_matrix = data_utils.parse_listener(code_scenarios, listener_file, adj_scenarios, codes_list, single_adj=True)
    listener_entropy = get_data_entropy(listener_matrix, num_dim=3)
    response_keys, response_vals = get_response_distribution(listener_matrix, num_scenarios=len(code_scenarios))
    mean = sum([response_vals[i]*response_keys[i] for i in range(len(response_vals))])
    create_histogram(listener_entropy, '../img/listener_uniform_histogram.pdf')
    create_cdf(listener_entropy, '../img/listener_uniform_cdf.pdf', response_keys, response_vals, NUM_NPAIR)
    print("mean", np.mean(listener_entropy))
    print("variance", np.var(listener_entropy))

