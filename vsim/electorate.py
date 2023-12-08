"""
Module for generating different kinds of electorate, represented
as a (n_individuals, n_issues) matrix.
"""
import numpy as np
from functools import partial
from sklearn.datasets import make_blobs
from typing import Dict, Callable, Optional


def generate_polarized_electorate(
    electorate_size: int,
    issues: int,
    clusters: int,
    seed: Optional[int] = None,
    cluster_std=1,
) -> np.ndarray:
    """Generates a voter base which is polarized, i.e. split in distinct clusters"""
    return np.array(
        make_blobs(
            n_samples=electorate_size,
            n_features=issues,
            centers=clusters,
            random_state=seed,
            cluster_std=cluster_std,
        )[0]
    )

def generate_swedish_electorate( # generate swedish voter base
    electorate_size: int,
    issues: int,
   # clusters: int,
    seed: Optional[int] = None,
    cluster_std=1,
) -> np.ndarray:
    
    issues = 2 #ändra i framtiden
    parties = np.array(
    [
        [1.9411764, 1.7647059],
        [4.4117646, 4.1176472],
        [2.2352941, 8],
        [3.2352941, 7.1176472],
        [5.9411764, 7.7058825],
        [7.0588236, 7.2352943],
        [1.5882353, 3.9411764],
        [8.7647057, 5.5882354],
    ])


    """
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.
    """
    return np.array(
        make_blobs( 
            centers=parties,
            n_samples=electorate_size,
            n_features=issues,
            random_state=seed,
            cluster_std=0.5, # wtf vrf måste göra här
        )[0]
    )


ELECTORATE_SCENARIOS: Dict[str, Callable] = {
    "centered": partial(generate_polarized_electorate, clusters=1),
    "bipolar": partial(generate_polarized_electorate, clusters=2),
    "tripolar": partial(generate_polarized_electorate, clusters=3),
    "sweden": partial (generate_swedish_electorate)
}


def normalize(v: np.ndarray) -> np.ndarray:
    """Needs to happen for our distance metric to be comparable across systems"""
    return v / (np.linalg.norm(v) + 1e-16)


def setup_electorate(
    electorate_size, issues, scenario, seed: Optional[int] = None, cluster_std=1
) -> np.ndarray:
    return ELECTORATE_SCENARIOS[scenario](
            electorate_size, issues, seed=seed, cluster_std=cluster_std)
        
        #normalize(
        #ELECTORATE_SCENARIOS[scenario](
        #    electorate_size, issues, seed=seed, cluster_std=cluster_std
        #)
    #)


if __name__ == "__main__":
    pass
