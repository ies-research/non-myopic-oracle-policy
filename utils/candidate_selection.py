import numpy as np
from random import Random


def select_one_candidate(cand_idx_sets, mapping, id_clf, y_true, idx_eval,
                         cost_matrix, lookahead, random_seed, mode=None):
    rng = np.random.default_rng(seed=random_seed)

    # Storage for computed accuracy per candidate sample
    risk_table = np.full((len(mapping), lookahead), np.inf)

    for cand_idx_set in cand_idx_sets:
        m = len(cand_idx_set)
        mapped_idx_set = [mapping[x] for x in cand_idx_set]
        id_clf.partial_fit(mapped_idx_set, y=y_true[mapped_idx_set],
                           use_base_clf=True, set_base_clf=False)
        y_pred = id_clf.predict(idx_eval)
        risk = _risk_estimation(y_true[idx_eval], y_pred, cost_matrix, np.ones_like(y_pred))

        for cand_idx in cand_idx_set:
            risk_table[cand_idx, m - 1] = min(risk_table[cand_idx, m - 1], risk)

    if mode is None or mode == 'rand':
        # Calculate min risk of every candidate
        min_risk_table = np.min(risk_table, axis=1)
        # Only consider candidates with minimal min risk
        query_candidates = np.where(min_risk_table == np.min(min_risk_table))[0]
    elif mode == 'mean':
        # Calculate best lookahead for every candidate
        best_lookahead = np.argmin(risk_table, axis=1)
        # Calculate mean risk of every candidate
        mean_risk = np.array([np.mean(risk_table[i, :idx+1]) for i, idx in enumerate(best_lookahead)])
        query_candidates = np.where(mean_risk == min(mean_risk))[0]
    elif mode in ['ascending', 'descending']:
        # Calculate min risk of every candidate
        min_risk_table = np.min(risk_table, axis=1)
        # Only consider candidates with minimal min risk
        query_candidates = np.where(min_risk_table == np.min(min_risk_table))[0]
        # Go through all lookaheads in ascending/descending order
        if mode == 'ascending':
            lookaheads = range(risk_table.shape[1])
        else:
            lookaheads = range(risk_table.shape[1])[::-1]
        for i in lookaheads:
            # Calculate the risk of every remaining candidate for the current lookahead
            risk = risk_table[query_candidates, i]
            min_risk = np.min(risk)
            query_candidates = query_candidates[np.where(risk == min_risk)[0]]
    else:
        raise ValueError("If mode is not None, it must be in ['rand', 'mean', 'ascending', 'descending'].")
    mapped_query_candidates = mapping[query_candidates]
    query_index = rng.choice(mapped_query_candidates, size=1)
    return query_index


def select_batch(cand_idx_sets, mapping, id_clf, y_true, idx_eval, cost_matrix,
                 random_seed, mode=None):
    min_risk = np.inf
    best_batches = None

    # Evaluate every candidate set
    for cand_idx_set in cand_idx_sets:
        mapped_idx_set = [mapping[x] for x in cand_idx_set]
        id_clf.partial_fit(mapped_idx_set, y=y_true[mapped_idx_set],
                           use_base_clf=True, set_base_clf=False)
        y_pred = id_clf.predict(idx_eval)
        risk = _risk_estimation(y_true[idx_eval], y_pred, cost_matrix, np.ones_like(y_pred))

        if risk < min_risk:
            min_risk = risk
            best_batches = [mapped_idx_set]
        elif risk == min_risk:
            best_batches.append(mapped_idx_set)

    # Shuffle best batches and sort according to the selected method
    Random(random_seed).shuffle(best_batches)
    if mode is None or mode == 'rand':
        best_batch = best_batches[0]
    elif mode == 'ascending':
        best_batch = sorted(best_batches, key=len)[0]
    elif mode == 'descending':
        best_batch = sorted(best_batches, key=len, reverse=True)[0]
    else:
        raise ValueError("If mode is not None, it must be in ['rand', 'ascending', 'descending'].")
    return best_batch


def _risk_estimation(y_true, y_pred, cost_matrix, sample_weight):
    cost_est = cost_matrix[y_true, :][range(len(y_true)), y_pred]
    return np.sum(sample_weight * cost_est)
