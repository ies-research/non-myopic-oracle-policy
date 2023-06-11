from skactiveml.pool import RandomSampling, UncertaintySampling, \
    QueryByCommittee, ProbabilisticAL, EpistemicUncertaintySampling, DiscriminativeAL, Quire

from utils.optimal_strategy import Optimal
from utils.simulated_annealing_search import SimulatedAnnealingSearch


def get_query_strategy(algo_name, seed, classes):
    if algo_name == 'rand':
        return RandomSampling(random_state=seed)
    elif algo_name == 'lc':
        return UncertaintySampling(method='least_confident', random_state=seed)
    elif algo_name == 'epis':
        return EpistemicUncertaintySampling(random_state=seed)
    elif algo_name == 'quire':
        return Quire(classes, random_state=seed)
    elif algo_name == 'discriminative':
        return DiscriminativeAL(random_state=seed)
    elif algo_name == 'xpal':
        # TODO add xpal
        raise ValueError()
    elif algo_name == 'qbc':
        return QueryByCommittee(random_state=seed)
    elif algo_name == 'mcpal':
        return ProbabilisticAL(m_max=2, random_state=seed)
    elif algo_name == 'optimal_greedy':
        return Optimal(random_state=seed, nonmyopic_look_ahead=1)
    elif 'optimal' in algo_name:
        algo_split = algo_name.split('_')
        if len(algo_split) == 4:
            _, lookahead, batch, sample = algo_split
            selection_mode = None
        else:
            _, lookahead, batch, sample, selection_mode = algo_split
        sample = int(sample)
        lookahead = int(lookahead)
        return Optimal(random_state=seed, nonmyopic_look_ahead=lookahead,
                       sample=sample, sample_mode='std',
                       selection_mode=selection_mode,
                       allow_smaller_batch_size=True)
    elif 'simulated-annealing' in algo_name:
        num_sa_samples = int(algo_name.split('_')[1]) if len(algo_name.split('_')) > 1 else 25000
        num_greedy_samples = int(algo_name.split('_')[2]) if len(algo_name.split('_')) > 2 else 5000
        return SimulatedAnnealingSearch(
            random_state=seed, num_sa_samples=num_sa_samples, num_greedy_samples=num_greedy_samples
        )
    else:
        raise ValueError('algo_name {} not defined'.format(algo_name))
