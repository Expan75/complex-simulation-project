"""
Module for generating different kinds of electorate, represented
as a (n_individuals, n_issues) matrix.
"""
import numpy as np
from dataclasses import dataclass


def generate_random_population(population_size: int, issues: int) -> np.ndarray:
    """Generates a completely random voter base, without any tendency"""
    return np.random.rand(population_size, issues)


def generate_centered_population(population_size: int, issues: int) -> np.ndarray:
    """Generates a voter base with centrist tendencies"""
    return np.random.normal(loc=0, scale=1.0, size=(population_size, issues))


def generate_polarized_population(population_size: int, issues: int) -> np.ndarray:
    """Generates a voter base which is highly polarized"""
    # each issue has the population divided
    for i in range(1, issues + 1):
        pass

    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # verify we do indeed get distributions that make sense
    population, issues = (10_000, 2)

    pass
