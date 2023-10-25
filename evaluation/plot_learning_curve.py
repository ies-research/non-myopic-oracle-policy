import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from utils.evaluation import load_results

# Parse arguments
parser = argparse.ArgumentParser(description='Plot learning curves.')
parser.add_argument('--path', dest='path', type=str, default='results',
                    help='the path to store the evaluation output files')
parser.add_argument('--use_train_error', dest='use_train_error', action='store_const', const=True, default=False)
args = parser.parse_args()

plot_style = {
    'rand': {'label': 'Rand', 'color': '#858585', 'linestyle': '--'},
    'lc': {'label': 'LC', 'color': '#6DB356', 'linestyle': '--'},
    'qbc': {'label': 'QBC', 'color': '#21780F', 'linestyle': '--'},
    'epis': {'label': 'Epis', 'color': '#BB31B7', 'linestyle': '--'},
    'mc': {'label': 'mc', 'color': '#CBB13B', 'linestyle': '--'},
    'chap': {'label': 'chap', 'color': '#B4723F', 'linestyle': '--'},
    'voi': {'label': 'voi', 'color': '#9A0D0D', 'linestyle': '--'},
    'mcpal': {'label': 'PAL', 'color': '#291468', 'linestyle': '--'},
    'quire': {'label': 'QUIRE', 'linestyle': '--'},
    'discriminative': {'label': 'DAL', 'linestyle': '--'},
    'simulated-annealing': {'label': 'SA search'},
    'simulated-annealing_2000_400': {'label': 'SA fast'},
    # Style for optimal strategy
    'optimal_1_1_1000': {'label': '$M = 1, S = 1000$'},
    'optimal_2_1_500_ascending': {'label': '$M = 2, S = 500$'},
    'optimal_3_1_333_ascending': {'label': '$M = 3, S = 333$'},
    'optimal_4_1_250_ascending': {'label': '$M = 4, S = 250$'},
    'optimal_5_1_200_ascending': {'label': '$M = 5, S = 200$'},
    'optimal_2_1_200_ascending': {'label': '$M = 2, S = 200$'},
    'optimal_2_1_1000_ascending': {'label': '$M = 2, S = 1000$'},
    'optimal_5_1_1000_ascending': {'label': '$M = 5, S = 1000$'},
}

algorithms = dict()
algorithms['ablation_study'] = [
    'optimal_1_1_1000',
    'optimal_2_1_200_ascending',
    'optimal_2_1_500_ascending',
    'optimal_3_1_333_ascending',
    'optimal_4_1_250_ascending',
    'optimal_5_1_200_ascending',
    'optimal_2_1_1000_ascending',
    'optimal_5_1_1000_ascending',
]
algorithms['comparison'] = [
    'rand', 'lc', 'qbc', 'epis', 'mcpal', 'quire', 'discriminative',
    'simulated-annealing',
    'simulated-annealing_2000_400',
    'optimal_2_1_200_ascending',
    'optimal_5_1_1000_ascending',
]

datasets = [814, 1510, 54, 1462, 20, 36, 182, 11, 37, 1063, 1464, 1494, 1504]
for key, value in algorithms.items():
    results = load_results(args.path, datasets, value)
    for dataset_id in datasets:
        plt.rcParams.update({'font.size': 5})
        fig, ax = plt.subplots(figsize=(2.24, 1.44))
        ylim_top = 0
        for i_qs, qs_name in enumerate(value):
            if qs_name == 'epis' and dataset_id in [54, 20, 36, 182, 11]:
                continue
            n_labels = np.loadtxt(args.path + f'/{dataset_id}/n_labels#{qs_name}#{0}.csv')
            error = np.empty((25, len(n_labels)))
            time = []
            for seed in range(25):
                t = np.loadtxt(args.path + f'/{dataset_id}/time#{qs_name}#{seed}.csv')
                time.append(t)
                if args.use_train_error:
                    error[seed] = results[dataset_id][qs_name]['train_error'][seed]
                else:
                    error[seed] = results[dataset_id][qs_name]['test_error'][seed]
            mean_error = error.mean(axis=0)
            ylim_top = max(ylim_top, max(mean_error[2:]))
            ax.plot(n_labels, mean_error, lw=.5, **plot_style[qs_name])

        ylim_bottom = max(0, plt.ylim()[0])
        plt.ylim(ylim_bottom, ylim_top)
        data = fetch_openml(data_id=dataset_id, parser='auto')
        dataset_name = data['details']['name']

        ax.legend()
        # ax.set_title(dataset_name)
        ax.set_xlabel("# selected instances")
        ax.set_ylabel("Risk")
        if not args.use_train_error:
            os.makedirs('plots/learning_curves/', exist_ok=True)
            filename = f'plots/learning_curves/{key}_{dataset_id}.pdf'
        else:
            os.makedirs('plots/learning_curves_validation_error/', exist_ok=True)
            filename = f'plots/learning_curves_validation_error/{key}_{dataset_id}.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
