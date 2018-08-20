import rsa_model
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bargraph(hist_data, cat_labels, overall_name, scenario_name, sort_hist='mturk'):
    '''
    Saves generated histogram plot in plots folder
    :param hist_data: dictionary of distribution
    :param cat_labels: label of each category
    :param overall_name: str of scenario name
    :param scenario_name: str describing scenario details
    :param sort_hist:
    :return: None
    '''
    # sort categoric label order by mturk values
    sort_order = np.argsort(hist_data[sort_hist])[::-1]

    # convert into a single dataset
    d = {'item': [], "dist": [], 'dset': []}
    for k, val in hist_data.items():
        d['item'].extend(np.array(cat_labels)[sort_order])
        d['dist'].extend(np.array(val)[sort_order])
        d['dset'].extend([k for _ in range(len(val))])
    df = pd.DataFrame(data=d)

    # plot that dataset as barplot
    fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
    ax = sns.barplot(x="item", y="dist", hue="dset", data=df)
    sns.despine()
    ax.set(xlabel='', ylabel='')
    ax.legend().set_title('')
    ax.set(ylim=(0, 1.0))
    plt.xticks(rotation=90, ha='right')
    plt.savefig('../plots/{}/{}.png'.format(overall_name, scenario_name), bbox_inches="tight")
    plt.clf()
    plt.close('all')


def viz_raw_prob(num, adj_scen, code_list, word_list, adj_feats_list, feat_names_arr, comparison_str='listener_semantic'):
    fig, axes = plt.subplots(nrows=1, ncols=len(adj_feats_list), sharey=True)

    out_group = [code_list[1][i] for i in range(len(code_list[1]))]
    in_group = [code_list[0][0], code_list[0][1]]
    sample_list = in_group + out_group
    code_words = [word_list[n] for n in sample_list]
    title_str = "Scenario %d Normalized Probabilities"%num
    plt.yticks(np.arange(len(adj_scen)), adj_scen)

    for i, ax in enumerate(axes.flat):
        raw = rsa_model.raw_scenario_table(adj_scen, sample_list, adj_feats_list[i])
        ax.set_title(feat_names_arr[i])
        im = ax.imshow(raw, cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.sca(ax)
        plt.xticks(np.arange(len(code_words)), code_words, rotation=90)

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2)
    cax = fig.add_axes([0.92, 0.3, 0.01, 0.45])
    plt.colorbar(im, cax=cax, aspect=100)

    fig.text(0.55, 0.15,'Codenames (nouns)', ha='center')
    fig.text(0.04, 0.55, 'Clues (adjectives)', va='center', rotation='vertical')

    plt.suptitle(title_str)
    plt.savefig('../img/' + comparison_str + '/raw_scenario%d'%num)
    plt.close()


def viz_prag_prob(num, adj_scen, code_list, word_list, mat_dict, mat_str_list, comp='listener'):
    fig, axes = plt.subplots(nrows=1, ncols=len(mat_dict.keys()), sharey=True, figsize=(9,7))

    out_group = [code_list[1][i] for i in range(len(code_list[1]))]
    in_group = [code_list[0][0], code_list[0][1]]
    sample_list = in_group + out_group
    noun_pairs = list(itertools.combinations(sample_list, 2))
    if comp == 'listener':
        plot_titles = ['$s_{p,a}$', '$L_0$', '$S_1$', '$L_1$']
    else:
        plot_titles = ['$s_{p,a}$', '$S_0$', '$L_1$', '$S_1$']
    np_words = [word_list[np[0]] + ' ' + word_list[np[1]] for np in noun_pairs]
    title_str = "Scenario%d RSA Matrices" % num
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(mat_dict[mat_str_list[i]], cmap=plt.cm.binary, vmin=0, vmax=1)
        ax.set_title(plot_titles[i])
        plt.sca(ax)
        plt.xticks(np.arange(len(np_words)), np_words, rotation=90)

    plt.yticks(np.arange(len(adj_scen)), adj_scen)

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2)
    cax = fig.add_axes([0.92, 0.28, 0.01, 0.38])
    plt.colorbar(im, cax=cax, aspect=100)

    fig.text(0.09, 0.55, 'Clues (adjectives)', va='center', rotation='vertical')
    if comp == 'listener':
        fig.text(0.55, 0.15, 'Codename (noun) pairs answers', ha='center')
    else:
        fig.text(0.55, 0.55, 'Clues (adjectives) answers', ha='center')

    plt.suptitle(title_str)
    plt.savefig('../img/' + comp + '_pragmatic_geo/rsa_scenario%d'%num)

    plt.close()


def viz_scenario_speaker(num, model_prob, mturk_prob, comp_list_str, adjs_set, code_set, comparison_str ='semantic'):

    colors = plt.cm.tab20(np.linspace(0, 0.5, 10))
    color_dict = {'Bigram': colors[0], 'Conceptnet': colors[2], 'Word2Vec': colors[4], 'LDA': colors[6], 'Human': colors[8],
                  'Bigram Literal': colors[0], 'Bigram Pragmatic': colors[1]}

    model_prob = np.concatenate((model_prob, mturk_prob.reshape(1, len(mturk_prob))))
    comp_list_label = comp_list_str + ["Human"]
    color_labels = [color_dict[comp_str] for comp_str in comp_list_label]

    index = np.arange(len(adjs_set))
    bar_width = 0.20
    x_labels = ["Answer: \n" + p for i, p in enumerate(adjs_set)]

    for i in range(len(comp_list_label)):
        rects1 = plt.bar(index + i*bar_width, model_prob[i], bar_width,
                         color=color_labels[i],
                         label=comp_list_label[i])

    plt.xlabel('Clue (Adjective) Answer')
    plt.ylabel('Answer probability')
    title_str = "Scenario %d \n" % num + "Target Codenames: %s, %s\n"%(code_set[0], code_set[1])  \
                + "Non-Target Codenames: " + code_set[2]
    plt.title(title_str)
    plt.xticks(index + bar_width, x_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/speaker_' + comparison_str + '/' + "Scenario%d"%num)
    plt.close()


def viz_scenario_listener(num, model_prob, mturk_prob, comp_list_str, adjs_set, code_pairs, comparison_str='semantic'):
    colors = plt.cm.tab20(np.linspace(0, 0.5, 10))
    color_dict = {'Bigram': colors[0], 'Conceptnet': colors[2], 'Word2Vec': colors[4], 'LDA': colors[6], 'Human': colors[8],
                  'Bigram Literal': colors[0], 'Bigram Pragmatic': colors[1]}

    model_prob = np.concatenate((model_prob, mturk_prob.reshape(1, len(mturk_prob))))
    comp_list_label = comp_list_str + ["Human"]
    print(comp_list_label)
    color_labels = [color_dict[comp_str] for comp_str in comp_list_label]

    x_labels = ["Answer: \n" + p[0] + ' ' + p[1] for i, p in enumerate(code_pairs)]

    index = np.arange(len(code_pairs))
    bar_width = 0.20
    for i in range(len(comp_list_label)):
        rects1 = plt.bar(index + i*bar_width, model_prob[i], bar_width, color = color_labels[i],
                         label=comp_list_label[i])


    plt.xlabel('Codename (Noun) Pair Answer')
    plt.ylabel('Answer probability')
    title_str = "Scenario %d \n"%num + "Clue given: %s \n"%adjs_set[0] + "Other clues: " + " , ".join(adjs_set[1:])
    plt.title(title_str)
    plt.xticks(index + bar_width, x_labels)
    plt.tight_layout()
    plt.legend()
    plt.savefig('../img/listener_' + comparison_str + '/' + "Scenario%d"%num)
    plt.close()