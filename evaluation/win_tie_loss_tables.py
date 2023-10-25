import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils.evaluation import load_results

dataset_ids = [814, 1510, 54, 1462, 20, 36, 182, 11, 37, 1063, 1464, 1494, 1504]

# List of all selection strategies to be evaluated
opt_methods = [
    'optimal_1_1_1000',
    'optimal_2_1_200_ascending',
    'optimal_4_1_250_ascending',
    'optimal_5_1_1000_ascending',
]

selection_methods = [
    'mcpal', 'qbc', 'lc', 'rand', 'quire', 'discriminative', 'epis',
    'simulated-annealing', 'simulated-annealing_2000_400'
]

names = {
    'optimal_greedy': 'greedy',
    'optimal_1_1_1000': 'opt 1/1000',
    'optimal_2_1_200_ascending': 'opt 2/200',
    'optimal_4_1_250_ascending': 'opt 4/250',
    'optimal_5_1_1000_ascending': 'opt 5/1000',
    # Competitors
    'mcpal': 'PAL',
    'qbc': 'QBC',
    'lc': 'LC',
    'rand': 'rand',
    'quire': 'QUIRE',
    'discriminative': 'DAL',
    'epis': 'epis',
    'simulated-annealing': 'SA search',
    'simulated-annealing_2000_400': 'SA fast',
}

pval = .05 / (len(selection_methods) * len(dataset_ids) * 2 - 6)

results_path = "results/"
results = load_results(results_path, dataset_ids, methods=opt_methods + selection_methods)

df = pd.DataFrame()
for opt_method in opt_methods:
    for selection_method in selection_methods:
        win = 0
        tie = 0
        loss = 0
        for ds_id in dataset_ids:
            non_binary = [54, 20, 36, 182, 11]
            if selection_method == 'epis' and ds_id in non_binary:
                continue
            x = np.nanmean(results[ds_id][selection_method]['test_error'], axis=1)
            y = np.nanmean(results[ds_id][opt_method]['test_error'], axis=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = wilcoxon(x, y, alternative='greater')
                p_greater = res.pvalue
                res = wilcoxon(x, y, alternative='less')
                p_less = res.pvalue
            if p_greater < pval:
                win += 1
            elif p_less < pval:
                loss += 1
            else:
                tie += 1
        win_tie_loss_str = f'{win} ' \
                            f'/ {tie} ' \
                            f'/ {loss}'
        df.loc[names[opt_method], names[selection_method]] = win_tie_loss_str

# Print LaTeX table
df = df.transpose()
print('Table for AULC')
print(df.style.to_latex(column_format='lcccc', hrules=True))

# Print LaTeX table for computation time
methods = opt_methods + selection_methods
methods.remove('epis')
# methods = ['epis', 'optimal_2_1_200_ascending', 'optimal_1_1_1000', 'optimal_2_1_500_ascending']
time_list = []
for method in methods:
    ds_id = 182
    time = np.mean(np.nansum(results[ds_id][method]['time'], axis=1))
    time_list.append(time)
df_time = pd.DataFrame(time_list, index=[names[x] for x in methods], columns=['time (s)'])
style = df_time.style.format(precision=1)
print(style.to_latex(hrules=True))

time_table = [[np.mean(np.nansum(results[ds_id][method]['time'], axis=1)) for ds_id in dataset_ids] for method in methods]
# import time
# time_table = [[time.strftime('%d-%H:%M:%S', time.gmtime(np.mean(np.nansum(results[ds_id][method]['time'], axis=1)))) for ds_id in dataset_ids] for method in methods]
df_time = pd.DataFrame(time_table, index=[names[x] for x in methods], columns=dataset_ids)
style = df_time.style.format(precision=1)
print(style.to_latex(hrules=True))
