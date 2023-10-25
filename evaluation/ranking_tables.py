import argparse

import numpy as np
import pandas as pd

from utils.evaluation import compute_time, load_results, compute_ranks


parser = argparse.ArgumentParser(description='Create ranking tables.')
parser.add_argument('--path', dest='path', type=str, default='results/',
                    help='the path where the result files are stored')
parser.add_argument('--loss', dest='loss', type=str, default='test_error')
parser.add_argument('--time_dataset_id', dest='time_dataset_id', type=int, default=182)
args = parser.parse_args()

if args.loss == 'test_error':
    print("Plot table for test error.")
elif args.loss == 'train_error':
    print("Plot table for train (evaluation) error.")


def print_dataframe(df, highlight_subset=None):
    df.index.rename([('Parameters', 'M'), ('Parameters', 'S'), ('Parameters', 'order')],
                    inplace=True)
    df.reset_index(inplace=True)
    style = df.style
    if highlight_subset is not None:
        style = style.highlight_min(
            subset=highlight_subset,
            props='color:{red}; bfseries:;'
        )
    style = style.format(precision=2)
    style = style.hide(axis="index")
    print(style.to_latex(hrules=True, column_format='c' * df.shape[1], multicol_align='c'))


dataset_ids = [814, 1063, 1510, 11, 1464, 37, 54, 1494, 1462, 1504, 20, 36, 182]
dataset_columns = [str(x) for x in dataset_ids]

methods = [
    # Greedy
    'optimal_1_1_1000',
    # Ascending
    'optimal_2_1_500_ascending',
    'optimal_3_1_333_ascending',
    'optimal_4_1_250_ascending',
    'optimal_5_1_200_ascending',
    # Descending
    'optimal_2_1_500_descending',
    'optimal_3_1_333_descending',
    'optimal_4_1_250_descending',
    'optimal_5_1_200_descending',
    # Other
    # 'optimal_2_1_200_descending',
    # 'optimal_2_1_500_descending',
    # 'optimal_2_1_1000_descending',
    # 'optimal_5_1_1000_descending',
]

# Create index
m_values = []
s_values = []
order_values = []
for method in methods:
    parameters = method.split('_')
    if len(parameters) == 4:
        _, M, _, S = parameters
        order = '--'
    else:
        _, M, _, S, order = parameters
    m_values.append(M)
    s_values.append(S)
    order_values.append(order)
order_values = list(map(lambda x: x.replace('descending', 'desc').replace('ascending', 'asc'), order_values))
index = [np.array(m_values), np.array(s_values), np.array(order_values)]

columns = [
    np.array(["20", "20", "50", "50", "mean", "mean"]),
    np.array(["rank", "time", "rank", "time", "rank", "time"]),
]

# Shape (n_methods, 2)
mean_data = np.zeros((len(methods), 2))

final_test_error = dict()
mean_test_error = dict()
final_time = {}
# Load all required results
results = load_results(args.path, dataset_ids, methods)
for dataset_id in dataset_ids:
    final_test_error[dataset_id] = dict()
    mean_test_error[dataset_id] = dict()
    final_time[dataset_id] = dict()
    for i_method, method in enumerate(methods):
        # Load results of shape (n_repetitions, budget + 1)
        a = results[dataset_id][method][args.loss]
        # Save mean test error over all iterations in array of shape (n_repetitions)
        mean_test_error[dataset_id][method] = np.nanmean(a, axis=1)
        # Save time required for querying all instances in array of shape (n_repetitions)
        a = results[dataset_id][method]['time']
        final_time[dataset_id][method] = np.nansum(a, axis=1)

# Create array for the ranks in all datasets together with mean rank
ranking_data = np.zeros((len(methods), len(dataset_ids)+1))

# Compute ranks array of shape (n_datasets, n_methods)
all_ranks = compute_ranks(methods, dataset_ids, mean_test_error)
mean_ranks = np.mean(all_ranks, axis=0)
ranking_data[0:len(methods), :-1] = all_ranks.T
ranking_data[0:len(methods), -1] = mean_ranks
computation_time = compute_time(methods, args.time_dataset_id, final_time)
mean_data[0:len(methods), 0] = mean_ranks
mean_data[0:len(methods), 1] = computation_time

# Plot table with every data set
df = pd.DataFrame(ranking_data, index=index, columns=dataset_columns + ['mean'])
df.index.rename(['M', 'S', 'order'], inplace=True)
df = df.reset_index()
s = df.style.highlight_min(
    subset=dataset_columns + ['mean'],
    props='color:{red}; bfseries:;'
)
s = s.format(precision=2)
s = s.hide(axis="index")
print(s.to_latex(hrules=True, column_format='c'*df.shape[1]))

# Overview table for AULC
df = pd.DataFrame(mean_data, index=index, columns=list(np.array(columns)[:, 4:]))
print_dataframe(df, highlight_subset=[['mean', 'rank']])
