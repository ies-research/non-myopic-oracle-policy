from os import path, listdir

import numpy as np
from scipy.stats import rankdata

from utils.dataset_utils import load_dataset


def compute_ranks(algorithms, dataset_ids, data):
    """Return ranks for all data sets for every strategy."""
    all_ranks = []
    for dataset_id in dataset_ids:
        tmp = [data[dataset_id][algo] for algo in algorithms]
        ranks = np.mean(rankdata(tmp, axis=0), axis=1)
        all_ranks.append(ranks)
    return np.array(all_ranks)


def compute_time(algorithms, dataset_id, data):
    tmp = [data[dataset_id][algo] for algo in algorithms]
    time = np.mean(tmp, axis=1)
    return time


def load_results(results_path, dataset_ids=None, methods=None):
    results = {}
    if dataset_ids is None:
        dataset_ids = np.array([61, 446, 796, 1523, 40671, 831, 1508, 455, 814, 451])
    for dataset_id in dataset_ids:
        results[dataset_id] = {}
        ds_path = path.join(results_path, str(dataset_id))

        files_names = \
            [f for f in listdir(ds_path) if path.isfile(path.join(ds_path, f))]
        data = [fn[:-4].split('#') for fn in files_names]
        _, available_methods, seeds = zip(*data)
        if methods is None:
            methods = np.unique(available_methods)
        seeds = np.unique(seeds)  # Array of all seeds

        X, y, dataset_name = load_dataset(dataset_id)

        # Load test error of all methods
        for i_method, method in enumerate(methods):
            if method == 'epis' and dataset_id in [54, 20, 36, 182]:
                continue
            results[dataset_id][method] = dict()
            for measure in ['test_error', 'train_error', 'time']:
                results[dataset_id][method][measure] = []
                for seed in seeds:
                    results[dataset_id][method][measure].append(
                        np.loadtxt(f'{ds_path}/{measure}#{method}#{seed}.csv')
                    )
                results[dataset_id][method][measure] = np.array(results[dataset_id][method][measure])
                if measure == 'test_error':
                    # TODO dont normalize?
                    results[dataset_id][method][measure] /= (len(X) * 0.4)
            # TODO batch does not work, fix
            results[dataset_id][method]['final_test_error'] = results[dataset_id][method]['test_error'][:, -1]
    return results
