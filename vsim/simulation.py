import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from vsim import common
from vsim.voting_system import VotingSystem, ElectionResult


@dataclass
class SimulationResult:
    measured_fairness: float
    election_result: ElectionResult
    parameters: Optional[dict] = None


class VotingSimulator:
    """
    Represents a running simulation. Runs an election given
    the injected voting system strategy and population.
    """

    def __init__(
        self,
        electorate: np.ndarray,
        candidates: np.ndarray,
        system: VotingSystem,
        plot: bool = False,
        seed: Optional[int] = None,
        log: Optional[logging.Logger] = None,
        scenario: Optional[str] = None,
    ):
        # misc settings
        np.random.seed(seed)  # None means random without seed
        self.plot = plot
        self.log = log if log is not None else common.conf_logger(1, "vsim.log")

        # sim params
        self.scenario = scenario if scenario is None else "no scenario"
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
        simulation_result = {
            "election_result": result,
            "measured_fairness": fairness,
            "parameters": {},
        }

        if self.plot:
            self.display(result, fairness)

        return SimulationResult(**simulation_result)


if __name__ == "__main__":
    pass
