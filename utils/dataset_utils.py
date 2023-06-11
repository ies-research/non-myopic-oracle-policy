import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(dataset_id, max_samples=None):
    dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser='liac-arff')
    filename = dataset.details['name']

    X = dataset.data
    y = LabelEncoder().fit(dataset.target).transform(dataset.target)

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    # Remove nans
    notnans = ~np.isnan(X).any(axis=1)
    X = X[notnans]
    y = y[notnans]

    return X, y, filename


def preprocess_dataset(X, y, train_size=0.8, random_state=None):
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def get_gamma_heuristic(X):  # TODO move to another file
    # Mean criterion by Chaudhuri et al. (2017)
    # The mean and median criteria for kernel bandwidth selection for support
    # vector data description
    delta = np.sqrt(2) * 1e-6
    N = X.shape[0]
    D = X.shape[1]
    s = np.sqrt((2 * N * D) / ((N - 1) * np.log((N - 1) / delta ** 2)))
    gamma = 1 / (2 * s ** 2)

    return gamma
