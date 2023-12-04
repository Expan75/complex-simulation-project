import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from src.vsim.voting_system import VotingSystem, ElectionResult


class VotingSimulator:
    """
    Represents a running simulation. Runs an election given
    the injected voting system strategy and population.
    """

    def __init__(
        self,
        electorate: np.ndarray,
        candidates: np.ndarray,
        scenario: str,
        log: logging.Logger,
        system: VotingSystem,
        plot: bool = False,
        seed: Optional[int] = None,
    ):
        # misc settings
        np.random.seed(seed)  # None means random without seed
        self.plot = plot
        self.log = log

        # sim params
        self.scenario = scenario
        self.voting_system: VotingSystem = system

        # simulation agents
        self.electorate: np.ndarray = electorate
        self.candidates: np.ndarray = candidates

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def n_voters(self) -> int:
        return int(self.electorate.shape[0])

    @property
    def n_issues(self) -> int:
        return int(self.electorate.shape[1])

    def calculate_fairness(self, result: ElectionResult) -> float:
        """Fairness is calculated as the average distance to the winner"""
        avg_distances = []
        for winner in result.winners:
            avg_dist_to_winner = np.mean(
                np.linalg.norm(self.candidates[winner] - self.electorate)
            )
            avg_distances.append(avg_dist_to_winner)

        return float(np.mean(avg_distances))

    def display(self, result: ElectionResult, fairness: float):
        """Renders an election"""
        self.log.debug("displaying")
        assert self.n_issues <= 2, "can only visualise 2D elections"
        _, ax = plt.subplots()

        columns = [f"issue_{i}" for i in range(1, self.n_issues + 1)]
        electorate_df = pd.DataFrame(self.electorate, columns=columns)
        electorate_df["state"] = "voter"

        # add candidates to same df to ease plotting
        candidate_df = pd.DataFrame(self.candidates, columns=columns)
        candidate_df["state"] = "candidate"
        df = pd.concat([electorate_df, candidate_df])

        sns.scatterplot(data=df, x="issue_1", y="issue_2", hue="state", ax=ax)
        ax.set_title(f"scenario={self.scenario}, {fairness=}")

        plt.show()

    def run(self):
        self.log.debug("running voting sim")
        result = self.voting_system.elect(self.electorate, self.candidates)
        fairness = self.calculate_fairness(result)
        print(f"fairness={fairness:.2f}")

        if self.plot:
            self.display(result, fairness)


if __name__ == "__main__":
    pass
