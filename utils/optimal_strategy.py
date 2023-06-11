import warnings
from copy import deepcopy

import numpy as np
from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool.utils import IndexClassifierWrapper
from skactiveml.utils import (
    check_type,
    check_cost_matrix,
    MISSING_LABEL,
    check_equal_missing_label,
)
from sklearn.metrics import pairwise_kernels

from utils.candidate_selection import select_one_candidate, select_batch
from utils.generate_cand_sets import (
    generate_all_sets,
    generate_full_sets,
    generate_sampled_sets,
    generate_sampled_sets_batch, sample_distance_based
)
from utils.help_functions import concatenate_samples


class Optimal(SingleAnnotatorPoolQueryStrategy):
    """An optimal strategy and different approximations.

    Parameters
    ----------
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Used for misclassification loss and ignored for log loss.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.
    nonmyopic_look_ahead : int, default=2
        TODO describe
    subtract_current : bool, default=False
        If true, the current error estimate is subtracted from the simulated
        score. This might be helpful to define a stopping criterion.
    method : str, default="misclassification_loss"
        The optimization method. Possible values are 'misclassification_loss'
        and 'log_loss'.
    """

    def __init__(
        self,
        cost_matrix=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        nonmyopic_look_ahead=2,
        subtract_current=False,
        method="misclassification_loss",
        allow_smaller_batch_size=False,
        sample=None,
        sample_mode=None,
        selection_mode=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cost_matrix = cost_matrix
        self.nonmyopic_look_ahead = nonmyopic_look_ahead
        self.allow_smaller_batch_size = allow_smaller_batch_size
        self.subtract_current = subtract_current
        self.method = method
        self.sample = sample
        self.sample_mode = sample_mode
        self.selection_mode = selection_mode

    def query(
        self,
        X,
        y,
        clf,
        candidates,
        fit_clf=True,
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
            Complete training data set including the labeled and unlabeled samples.
        y : array-like of shape (n_samples)
            All labels of the training data set including the labels of unlabeled instances.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        candidates : array-like of shape (n_candidates), dtype=int
            The indices of candidate instances in `X`.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        ignore_partial_fit : bool, default=True
            Relevant in cases where `clf` implements `partial_fit`. If True,
            the `partial_fit` function is ignored and `fit` is used instead.
        X_eval : array-like of shape (n_eval_samples, n_features), default=None
            Evaluation data set that is used for computing the risk.
        y_eval : array-like of shape (n_eval_samples), default=None
            Labels of the evaluation data set.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
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
        (
            X,
            y,
            clf,
            candidates,
            X_eval,
            batch_size,
            return_utilities,
        ) = self._validate_data(
            X,
            y,
            clf,
            candidates,
            X_eval,
            batch_size,
            return_utilities,
            reset=True,
            check_X_dict=None,
        )

        X_full, y_full, idx_train, idx_cand, idx_eval = concatenate_samples(X, y, candidates, X_eval, y_eval)

        # Set batch size to the nonmyopic look-ahead if query_batch is true
        if self.nonmyopic_look_ahead < batch_size:
            raise ValueError("self.nonmyopic_look_ahead must not be smaller than batch_size.")
        if batch_size > 1:
            nonmyopic_look_ahead = batch_size
        else:
            nonmyopic_look_ahead = self.nonmyopic_look_ahead

        # Check fit_clf
        check_type(fit_clf, "fit_clf", bool)

        # Initialize classifier that works with indices to improve readability
        y_known = y_full.astype(float)
        y_known[idx_cand] = self.missing_label
        id_clf = IndexClassifierWrapper(
            deepcopy(clf), X_full, y_known, set_base_clf=True,
            ignore_partial_fit=ignore_partial_fit, enforce_unique_samples=True,
            use_speed_up=False, missing_label=self.missing_label
        )

        # Fit the classifier.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            id_clf = self._precompute_and_fit_clf(id_clf, idx_train, idx_cand, idx_eval, fit_clf=fit_clf)

        # Check cost matrix.
        classes = id_clf.classes_
        self._validate_cost_matrix(len(classes))

        # Generate candidate sets
        candidates = X[idx_cand]
        cand_labels = y[idx_cand]
        if self.sample is None or self.sample == 1:
            if batch_size == 1 or self.allow_smaller_batch_size:
                cand_idx_sets = generate_all_sets(candidates, nonmyopic_look_ahead)
            else:
                cand_idx_sets = generate_full_sets(candidates, nonmyopic_look_ahead)
        elif self.sample_mode == 'std':
            if batch_size == 1 or self.allow_smaller_batch_size:
                cand_idx_sets = generate_sampled_sets(
                    candidates, nonmyopic_look_ahead, self.sample, random_seed=self.random_state
                )
            else:
                cand_idx_sets = generate_sampled_sets_batch(
                    candidates, nonmyopic_look_ahead, self.sample, random_seed=self.random_state
                )
        elif self.sample_mode == 'distance':
            if isinstance(clf, ParzenWindowClassifier):
                metric = clf.metric
                metric_dict = clf.metric_dict_
            else:
                metric = 'rbf'
                metric_dict = {}
            K = pairwise_kernels(
                candidates, metric=metric, **metric_dict
            )

            if batch_size == 1:
                cand_idx_sets = sample_distance_based(
                    candidates, nonmyopic_look_ahead, self.sample, random_seed=self.random_state, y=cand_labels,
                    metric=metric, metric_dict=metric_dict, K=K
                )
            else:
                raise ValueError("Batch-mode not supported for distance.")
        else:
            raise ValueError("std and distance are the only supported sample modes.")

        if batch_size == 1:
            best_idx = select_one_candidate(
                cand_idx_sets, idx_cand, id_clf, y_full, idx_eval,
                self.cost_matrix_, nonmyopic_look_ahead,
                self.random_state, mode=self.selection_mode
            )
        else:
            best_idx = select_batch(cand_idx_sets, idx_cand, id_clf, y_full,
                                    idx_eval, self.cost_matrix_,
                                    self.random_state, mode=self.selection_mode)

        return np.array(best_idx)

    def _validate_data(
        self,
        X,
        y,
        clf,
        candidates,
        X_eval,
        batch_size,
        return_utilities,
        reset=True,
        check_X_dict=None,
    ):

        # Validate input parameters.
        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=reset,
            check_X_dict=check_X_dict,
        )

        # Validate classifier type.
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label)

        self._validate_init_params()

        return (
            X,
            y,
            clf,
            candidates,
            X_eval,
            batch_size,
            return_utilities,
        )

    def _validate_init_params(self):
        """Function used to evaluate parameters of the `__init__` function that
        are not part of the abstract class to avoid redundancies.
        """
        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError(
                "{} is an invalid type for method. Type {} is "
                "expected".format(type(self.method), str)
            )
        if self.method not in ["misclassification_loss", "log_loss"]:
            raise ValueError(
                f"Supported methods are `misclassification_loss`, or"
                f"`log_loss` the given one is: {self.method}"
            )

        check_type(self.subtract_current, "subtract_current", bool)

        if self.method == "log_loss" and self.cost_matrix is not None:
            raise ValueError(
                "`cost_matrix` must be None if `method` is set to `log_loss`"
            )

    def _precompute_and_fit_clf(self, id_clf, idx_train, idx_cand, idx_eval, fit_clf):
        id_clf.precompute(idx_train, idx_cand)
        id_clf.precompute(idx_train, idx_eval)
        id_clf.precompute(idx_cand, idx_eval)
        if fit_clf:
            id_clf.fit(idx_train, set_base_clf=True)
        return id_clf

    def _validate_cost_matrix(self, n_classes):

        cost_matrix = (
            1 - np.eye(n_classes)
            if self.cost_matrix is None
            else self.cost_matrix
        )
        self.cost_matrix_ = check_cost_matrix(cost_matrix, n_classes)

    def _risk_estimation(
        self, y_true, y_pred, cost_matrix, sample_weight
    ):
        cost_est = cost_matrix[y_true, :][range(len(y_true)), y_pred]
        return np.sum(sample_weight * cost_est)

    def _logloss_estimation(self, prob_true, prob_pred):
        return -np.sum(prob_true * np.log(prob_pred + np.finfo(float).eps))