import itertools
import random
from copy import deepcopy

import numpy as np
from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from skactiveml.utils import check_type, MISSING_LABEL, check_equal_missing_label
from sklearn import clone
from tqdm import trange

from utils.help_functions import concatenate_samples


class SimulatedAnnealingSearch(SingleAnnotatorPoolQueryStrategy):
    """The simulated annealing search proposed by [1].

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.
    anneal_factor : float, default=0.1
    num_sa_samples : int, default=25000
    num_greedy_samples : int, default==5000
    """

    def __init__(self, missing_label=MISSING_LABEL, random_state=None, anneal_factor=0.1, num_sa_samples=25000,
                 num_greedy_samples=5000, tot_acq=50, number_of_cores=1):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.N_pool = 2000
        self.anneal_factor = anneal_factor
        self.num_sa_samples = num_sa_samples
        self.num_greedy_samples = num_greedy_samples
        self.tot_acq = tot_acq
        self.number_of_cores = number_of_cores
        self.query_order = None

    def query(
        self,
        X,
        y,
        clf,
        candidates,
        ignore_partial_fit=True,
        X_eval=None,
        y_eval=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL).
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        y_true : array-like of shape (n_samples)
            True labels of all instances in X without missing labels.
        ignore_partial_fit : bool, optional (default=True)
            Relevant in cases where `clf` implements `partial_fit`. If True,
            the `partial_fit` function is ignored and `fit` is used instead.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        X, y, clf, candidates, X_eval, batch_size, return_utilities, = self._validate_data(
            X, y, clf, candidates, X_eval, batch_size, return_utilities, reset=True, check_X_dict=None
        )

        X_full, y_full, idx_train, idx_cand, idx_eval = concatenate_samples(X, y, candidates, X_eval, y_eval)
        self.N_pool = len(idx_train)

        # Find order of instances if not already saved
        if self.query_order is None:
            y_known = y_full.astype(float)
            y_known[idx_cand] = self.missing_label
            self.query_order = self.compute_query_order(clf, X_full, y_full, idx_eval, batch_size)

        # Select the next instance of the computed permutation that is not yet labeled
        next_query_index = next(idx for idx in self.query_order if idx in candidates)
        return np.array(next_query_index)

    def compute_query_order(self, clf, X, y_true, idx_eval, batch_size):
        """Compute query order using Simulated Annealing Search.

        Parameters
        ----------
        clf
        X : array-like of shape (n_samples, n_features)
            Training data set including the labeled and unlabeled samples.
        y_true : array-like of shape (n_samples)
            True labels of all instances in X without missing labels.
        idx_eval : array-like of shape (n_candidates)
        batch_size : int
            The number of samples to be selected in one AL cycle.

        Returns
        -------
        order : np.ndarray of shape (self.N_pool)
            The computed query order.
        """
        # Start with a random permutation of the instances
        order = random.sample(range(self.N_pool), self.N_pool)
        curve, quality = self.evaluate_order(order, clf, X, y_true, idx_eval)
        best_quality = quality
        best_order = order

        for i in trange(self.num_sa_samples):
            T = (i + 1) * self.anneal_factor
            new_order = swap_kernel(order, self.tot_acq, batch_size)
            new_curve, new_quality = self.evaluate_order(new_order, clf, X, y_true, idx_eval)
            ratio = np.exp((new_quality - quality) * T)
            if random.random() < ratio:
                order = new_order
                quality = new_quality
                if quality > best_quality:
                    best_quality = quality
                    best_order = order
        order = best_order
        quality = best_quality

        for _ in trange(self.num_greedy_samples):
            new_order = swap_kernel(order, self.tot_acq, batch_size)
            new_curve, new_quality = self.evaluate_order(new_order, clf, X, y_true, idx_eval)
            if new_quality > quality:
                order = new_order
                quality = new_quality
        return order

    def evaluate_order(self, order, clf, X, y_true, idx_eval):
        arguments = itertools.product(range(1, len(order) + 1), [clf], [order], [y_true], [X], [idx_eval])
        if self.number_of_cores == 1:
            curve = map(loss, arguments)
            curve = list(curve)
        else:
            import multiprocessing
            with multiprocessing.Pool(self.number_of_cores) as pool:
                curve = pool.map(loss, arguments)
        curve = np.array(curve)
        quality = np.mean(curve)
        return curve, quality

    def _validate_data(
        self, X, y, clf, candidates, X_eval, batch_size, return_utilities, reset=True, check_X_dict=None
    ):
        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = super()._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=reset, check_X_dict=check_X_dict
        )

        # Validate classifier type
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label)

        return X, y, clf, candidates, X_eval, batch_size, return_utilities

def swap_kernel(order, tot_acq, batch_size, internal_swp_prob=0.5):
    order = deepcopy(order)
    N_pool = len(order)
    if random.random() < internal_swp_prob:
        # Randomly swap two instances from different batches
        i, j = random.sample(range(tot_acq), 2)
        while int(i / batch_size) == int(j / batch_size):
            i, j = random.sample(range(tot_acq), 2)
        order[i], order[j] = order[j], order[i]
    else:
        # Randomly swap one queried instance with one that is not queried
        i = random.randint(0, tot_acq - 1)
        j = random.randint(tot_acq, N_pool - 1)
        order[i], order[j] = order[j], order[i]
    return order


def loss(args):
    idx, clf, order, y_train, X, idx_eval = args
    train_idx = order[:idx]
    clf = clone(clf).fit(X[train_idx], y_train[train_idx])
    return clf.score(X[idx_eval], y_train[idx_eval])
