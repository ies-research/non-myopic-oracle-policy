import argparse
import multiprocessing
import os
from time import time

import numpy as np

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices, call_func
from sklearn import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import zero_one_loss

from utils.dataset_utils import load_dataset, preprocess_dataset, get_gamma_heuristic
from utils.query_strategies import get_query_strategy

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates Active Learning Algorithms for the given dataset from openml.org.')
    parser.add_argument('dataset_id', type=int, help='the dataset id according to openml.org')
    parser.add_argument('algorithms', type=str, nargs='+', default='xpal', help='the algorithms to be performed')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('--budget', dest='budget', type=int, default=None, help='the number of labels to be acquired')
    parser.add_argument('--path', dest='path', type=str, default='../results/', help='the path to store the evaluation output files')
    parser.add_argument('--parallelize', action='store_const', const=True, default=False, help='parallelize computation of AULC')
    parser.add_argument('--ignore_log', dest='ignore_log', action='store_const', const=True, default=False, help='ignores the log')
    parser.add_argument('--prior', dest='prior', type=float, default=0.001, help='prior for xpal')
    args = parser.parse_args()

    # Get dataset
    X, y_true, dataset_name = load_dataset(args.dataset_id)
    X_train, X_test, X_valid, y_train, y_test, y_valid = \
        preprocess_dataset(X, y_true, train_size=0.2, random_state=args.seed)
    gamma = get_gamma_heuristic(X)

    budget = args.budget
    if budget is None:
        budget = int(len(X_train) / 2)
    os.makedirs(args.path + '/{}'.format(args.dataset_id), exist_ok=True)

    if not args.ignore_log:
        print('Dataset:', dataset_name)
        print('Instances:', len(X))
        print('Classes:', np.unique(y_true))
        print('Inst per class:', np.bincount(y_true))
        print('Ratio:', max(np.bincount(y_true))/len(y_true))
        print()
        print('Algorithms:', args.algorithms)
        print('Seed:', args.seed)
        print('Budget:', args.budget)
        print('Path:', args.path)
        print()
        print('Prior', args.prior)
        print('PWC Gamma', gamma)

    # Get active learning strategies
    query_strategies = [(algo_name, get_query_strategy(algo_name, args.seed, classes=np.unique(y_true)))
                        for algo_name in args.algorithms]

    for i_qs, (qs_name, qs) in enumerate(query_strategies):
        if qs_name == 'epis' and len(np.unique(y_true)) > 2:
            print('skipped')
            continue
        if 'simulated-annealing' in qs_name:
            qs.tot_acq = budget
            if args.parallelize:
                # Determine number of cores
                if 'SLURM_CPUS_PER_TASK' in os.environ:
                    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
                else:
                    number_of_cores = multiprocessing.cpu_count()
                print(f'Use {number_of_cores} cores')
                qs.number_of_cores = number_of_cores

        y_active = np.full(len(y_train), MISSING_LABEL)

        clf = ParzenWindowClassifier(
            classes=np.unique(y_train),
            metric_dict={'gamma': gamma},
            random_state=args.seed
        )
        clf_prior = ParzenWindowClassifier(
            classes=np.unique(y_train),
            metric_dict={'gamma': gamma},
            class_prior=1.e-3,
            random_state=args.seed
        )
        ensemble = SklearnClassifier(
            BaggingClassifier(
                ParzenWindowClassifier(classes=np.unique(y_train),
                                       metric_dict={'gamma': gamma})
            ),
            classes=np.unique(y_train)
        )
        discriminator = clone(clf)

        # Create dictionary for results
        results = dict()
        results['n_labels'] = np.full([budget + 1], np.nan)
        results['test_error'] = np.full([budget + 1], np.nan)
        results['train_error'] = np.full([budget + 1], np.nan)
        results['time'] = np.full([budget + 1], np.nan)

        for iteration in range(budget + 1):
            # Check if budget is exhausted
            left_budget = budget - len(labeled_indices(y_active))
            if left_budget <= 0:
                break

            if 'simulated-annealing' not in qs_name:
                print('.', end='')
            if iteration > 0:
                candidates = unlabeled_indices(y_active)
                y_strategy = y_active
                if qs_name == 'mc':
                    qs_args = {'clf': clf, 'X_eval': X_train[candidates]}
                elif qs_name == 'chap':
                    qs_args = {'clf': clf_prior, 'X_eval': X_train[candidates]}
                elif qs_name == 'discriminative':
                    qs_args = {'clf': clf_prior, 'discriminator': discriminator}
                elif qs_name == 'optimal_greedy':
                    qs_args = {'clf': clf, 'batch_size': 1}
                elif 'simulated-annealing' in qs_name:
                    qs_args = {'clf': clf, 'batch_size': 1}
                    y_strategy = y_train
                elif 'optimal' in qs_name:
                    batch_size = int(qs_name.split('_')[2])
                    qs_args = {'clf': clf, 'batch_size': min(batch_size, left_budget)}
                    qs.nonmyopic_look_ahead = min(qs.nonmyopic_look_ahead, left_budget)
                    y_strategy = y_train
                else:
                    qs_args = {'clf': clf}

                t = time()
                bestIdx = call_func(qs.query,
                                    X=X_train, y=y_strategy, y_true=y_train, candidates=candidates, X_eval=X_valid,
                                    y_eval=y_valid, **qs_args, ensemble=ensemble, return_utilities=False)
                execution_time = time() - t
                results['time'][iteration] = execution_time

                y_active[bestIdx] = y_train[bestIdx]

            clf.fit(X_train, y_active)

            # Store measures
            n_labels = len(labeled_indices(y_active))
            results['n_labels'][iteration] = n_labels
            results['test_error'][iteration] = \
                zero_one_loss(y_test, clf.predict(X_test), normalize=False)
            results['train_error'][iteration] = \
                zero_one_loss(y_valid, clf.predict(X_valid), normalize=False)
        print(f"\nExecution time: {np.nansum(results['time']):.2f}")

        print()
        for key in results.keys():
            np.savetxt(args.path + f'/{args.dataset_id}/{key}#{qs_name}#{args.seed}.csv', results[key])
