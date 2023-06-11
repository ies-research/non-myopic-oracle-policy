import itertools

import numpy as np
from scipy.special import comb
from sklearn.metrics import pairwise_kernels


def generate_all_sets(candidates, look_ahead):
    cand_idx_sets = []
    for m in range(1, look_ahead + 1):
        # Generate all candidate sets of length m
        cand_indices = np.arange(len(candidates))
        cand_idx_sets.extend(list(itertools.combinations(cand_indices, m)))
    # cand_idx_sets = np.array(cand_idx_sets)
    return cand_idx_sets


def generate_full_sets(candidates, look_ahead):
    cand_indices = np.arange(len(candidates))
    cand_idx_sets = list(itertools.combinations(cand_indices, look_ahead))
    # cand_idx_sets = np.array(cand_idx_sets)
    return cand_idx_sets


def generate_sampled_sets(candidates, look_ahead, fraction, random_seed=None):
    rng = np.random.default_rng(seed=random_seed)

    sampled_sets = []
    for m in range(1, look_ahead + 1):
        total_combinations = comb(len(candidates), m)
        n_candidates = _get_n_candidates(total_combinations, fraction)

        # Generate all candidate sets of length m
        if m == 1:
            # Draw from all candidates in the first iteration
            cand_indices = np.arange(len(candidates))
        else:
            # Only consider candidates that were already drawn in the last iteration
            cand_indices = np.unique(cand_idx_sets)

        cand_idx_sets = []
        while len(cand_idx_sets) < n_candidates:
            # TODO use hash value for cand set
            new_cand_idx_set = sorted(rng.choice(cand_indices, m, replace=False))
            if new_cand_idx_set not in cand_idx_sets:
                cand_idx_sets.append(new_cand_idx_set)

        # Sample the given fraction of candidates
        sampled_sets.extend(cand_idx_sets)
    return sampled_sets


def generate_sampled_sets_batch(candidates, look_ahead, fraction, random_seed=None):
    rng = np.random.default_rng(seed=random_seed)

    total_combinations = comb(len(candidates), look_ahead)
    n_candidates = _get_n_candidates(total_combinations, fraction)

    # Generate all candidate sets of length m
    cand_indices = np.arange(len(candidates))
    cand_idx_sets = []
    while len(cand_idx_sets) < n_candidates:
        new_cand_idx_set = list(rng.choice(cand_indices, look_ahead, replace=False))
        if new_cand_idx_set not in cand_idx_sets:
            cand_idx_sets.append(new_cand_idx_set)

    return cand_idx_sets


def sample_distance_based(candidates, look_ahead, fraction, random_seed, y, metric, metric_dict, K=None):
    # Calculate pairwise distance between all instances if not set
    # if distance is None:
    #     distance = np.empty((len(candidates), len(candidates)))
    #     for i, x1 in enumerate(candidates):
    #         for j, x2 in enumerate(candidates):
    #             distance[i, j] = np.linalg.norm(x1 - x2)
    # distance = np.array(distance)
    # if np.max(distance) == 0:
    #     distance = np.ones_like(distance)
    # distance /= np.max(distance)

    if K is None:
        K = pairwise_kernels(
            candidates, metric=metric, **metric_dict
        )
    # for i in range(len(candidates)):
    #     for j in range(len(candidates)):
    #         if

    cand_idx_sets = []
    for m in range(1, look_ahead + 1):
        # Generate all candidate sets of length m
        cand_indices = np.arange(len(candidates))
        all_sets = list(itertools.combinations(cand_indices, m))
        total_combinations = len(all_sets)
        n_candidates = _get_n_candidates(total_combinations, fraction)

        if n_candidates == total_combinations:
            cand_idx_sets.extend(np.array(all_sets))
            continue

        weights = np.ones(len(all_sets), dtype=float)
        for set_idx, cand_idx_set in enumerate(all_sets):
            weight = 1
            for i in range(len(cand_idx_set) - 1):
                for j in range(i + 1, len(cand_idx_set)):
                    idx1 = cand_idx_set[i]
                    idx2 = cand_idx_set[j]
                    if y[idx1] == y[idx2]:
                        weight *= K[idx1, idx2]
                    else:
                        weight *= 1 - K[idx1, idx2]
            weights[set_idx] = weight

        best_indices = np.argsort(weights)[:n_candidates]
        cand_idx_sets.extend(np.array(all_sets)[best_indices])
    return cand_idx_sets


def _get_n_candidates(n_instances, fraction):
    if isinstance(fraction, int):
        if not fraction >= 1:
            raise ValueError("If fraction is an int, is must be >= 1.")
        sampled_candidates = min(fraction, n_instances)
    elif isinstance(fraction, float):
        if not 0 < fraction <= 1:
            raise ValueError("If fraction is a float, it bust be in the interval (0, 1].")
        sampled_candidates = int(n_instances * fraction)
    else:
        raise TypeError("fraction must be of type int or float.")
    return sampled_candidates
