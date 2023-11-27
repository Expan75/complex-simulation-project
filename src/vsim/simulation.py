import logging
import numpy as np
from typing import Optional
from voting_system import VotingSystem, ElectionResult


__all__ = ["VotingSimulator"]


class VotingSimulator:
    """
    Represents a running simulation
    Runs an election given the injected voting system strategy
    """

    def __init__(
        self,
        log: logging.Logger,
        system: VotingSystem,
        seed: Optional[int] = None,
        population_size: int = 100,
        n_candidates: int = 2,
    ):
        np.random.seed(seed)  # None means random without seed
        self.log = log

        # sim params
        self.voting_system: VotingSystem = system
        self.voters: int = population_size
        self.candidates: int = n_candidates
        self.issues: int = 2

    def generate_electorate(self) -> np.ndarray:
        self.log.debug(f"generating electorate of size {(self.voters,self.issues)}")
        return np.random.rand(self.voters, self.issues)

    def generate_candidates(self) -> np.ndarray:
        self.log.debug(
            f"generating candidates of size {(self.candidates, self.issues)}"
        )

        return np.random.rand(self.candidates, self.issues)

    def calculate_fairness(
        self, voters: np.ndarray, candidates: np.ndarray, result: ElectionResult
    ) -> float:
        """Fairness is calculated as the average distance to the winner"""
        avg_distances = []
        for winner in result.winners:
            avg_dist_to_winner = np.mean(np.linalg.norm(candidates[winner] - voters))
            avg_distances.append(avg_dist_to_winner)

        return float(np.mean(avg_distances))

    def run_election(self):
        voters = self.generate_electorate()
        candidates = self.generate_candidates()
        result = self.voting_system.elect(electorate=voters, candidates=candidates)
        self.log.debug("election result")
        print(result)

        fairness = self.calculate_fairness(voters, candidates, result)
        print(f"fairness={fairness:.2f}")

    def run(self):
        self.log.debug("running voting sim")
        self.run_election()
        """Runs the simulation"""
