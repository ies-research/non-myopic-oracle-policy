import numpy as np


def concatenate_samples(X, y, candidates, X_eval, y_eval):
    X_full = X
    y_full = y
    idx_train = np.arange(len(X))
    idx_cand = candidates

    if X_eval is None:
        idx_eval = idx_train
    else:
        if y_eval is None:
            raise ValueError("If `X_eval` is specified, `y_eval` must be specified as well.")
        X_full = np.concatenate([X_full, X_eval], axis=0)
        y_full = np.concatenate([y_full, y_eval], axis=0)
        idx_eval = np.arange(len(X_full) - len(X_eval), len(X_full))

    return X_full, y_full, idx_train, idx_cand, idx_eval
