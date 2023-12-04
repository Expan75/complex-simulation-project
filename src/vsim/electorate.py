"""
Module for generating different kinds of electorate, represented
as a (n_individuals, n_issues) matrix.
"""
import numpy as np


def generate_random_electorate(electorate_size: int, issues: int) -> np.ndarray:
    """Generates a completely random voter base, without any tendency"""
    return np.random.rand(electorate_size, issues)


def generate_centered_electorate(electorate_size: int, issues: int) -> np.ndarray:
    """Generates a voter base with centrist tendencies"""
    return np.random.normal(loc=0, scale=1.0, size=(electorate_size, issues))


def generate_polarized_electorate(electorate_size: int, issues: int) -> np.ndarray:
    """Generates a voter base which is highly polarized"""
    partial_size = (electorate_size // 2, issues)
    left = np.random.normal(0 - 5, scale=1.0, size=partial_size)
    right = np.random.normal(0 + 5, scale=1.0, size=partial_size)
    electorate = np.r_[left, right]

    assert electorate.shape == (electorate_size, issues)
    return electorate

ELECTORATE_SCENARIOS = {
    "random": generate_random_electorate,
    "centered": generate_centered_electorate,
    "polarized": generate_polarized_electorate,
}


def setup_electorate(electorate_size, issues, scenario="random") -> np.ndarray:
    try:
        return ELECTORATE_SCENARIOS[scenario](electorate_size, issues)
    except KeyError:
        raise KeyError(f"{scenario} is not one of {set(ELECTORATE_SCENARIOS.keys())}")


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # verify we do indeed get distributions that make sense
    electorate, issues = (10_000, 2)
    columns = columns = [f"issue_{i}" for i in range(1, issues + 1)]

    # setup different electorates
    random_electorate = generate_random_electorate(electorate, issues)
    random_df = pd.DataFrame(random_electorate, columns=columns)

    centered_electorate = generate_centered_electorate(electorate, issues)
    centered_df = pd.DataFrame(centered_electorate, columns=columns)

    polarized_electorate = generate_polarized_electorate(electorate, issues)
    polarized_df = pd.DataFrame(polarized_electorate, columns=columns)



    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True)
    sns.kdeplot(random_df, ax=ax1)
    ax1.set_title("Randomised electorate with no opinion bias")

    sns.kdeplot(centered_df, ax=ax2)
    ax2.set_title("Centered electorate with few extreme opinions")

    sns.kdeplot(polarized_df, ax=ax3)
    ax3.set_title("Polarized electorate")

    plt.show()
