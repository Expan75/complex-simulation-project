import os
import sys
import json
import scipy
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

VOTING_SYSTEMS = {"plurality-with-runoff"}


parser = argparse.ArgumentParser("vsim", description="Voting simulator 0.0.1")
parser.add_argument("--voting-system", "-v", choices=VOTING_SYSTEMS)
parser.add_argument("--seed", "-s", type=int, default=None)
parser.add_argument("--log", "-l", type=str, default="DEBUG", required=False)
parser.add_argument("--plot", "-p", action="store_true", required=False)
parser.add_argument("--results-filepath", "-f", type=str, required=False)
args = parser.parse_args()


def conf_logger(
    loglevel: int,
    filepath: str,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    folder_path = ""
    for folder in filepath.split("/")[:-1]:
        folder_path = os.path.join(folder_path, folder)
        os.makedirs(folder, exist_ok=True)
    logging.basicConfig(filename=filepath, encoding="utf-8", level=loglevel, format=fmt)

    # add stdout logging too
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    log = logging.getLogger(__name__)
    log.addHandler(handler)

    return log


@dataclass
class ElectionResult:
    cast_votes: dict


class VotingSystem(ABC):
    """Strategy for running simulator, akin to given system of voting"""

    def __init__(self, params: dict):
        pass

    @abstractmethod
    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        pass


class NaivePlurality(VotingSystem):
    def __init__(self, params: dict):
        self.parmas = params

    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)
            preferred_candidate = np.argmin(distance)
            electoral_vote_count[preferred_candidate] += 1

        result = ElectionResult(electoral_vote_count)
        return result


class VotingSim:
    """Singleton for running the simulation"""

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

    def parse_result(self, results: ElectionResult) -> pd.DataFrame:
        outcome_df = pd.DataFrame(asdict(results))
        outcome_df["candidate"] = outcome_df.index
        outcome_df.columns = ["votes", "candidate"]
        return outcome_df.reset_index().drop(["index"], axis=1)

    def run_election(self):
        voters = self.generate_electorate()
        issues = self.generate_candidates()
        result = self.voting_system.elect(voters, issues)
        outcome = self.parse_result(result)
        self.log.debug("election result")
        print(outcome)

    def run(self):
        self.log.debug("running voting sim")
        self.run_election()
        """Runs the simulation"""


def setup_voting_system(choice: str) -> VotingSystem:
    if choice not in VOTING_SYSTEMS:
        raise RuntimeError(f"{choice=} not one of supported {VOTING_SYSTEMS=}")
    else:
        return NaivePlurality(params={})


if __name__ == "__main__":
    args = parser.parse_args()

    # setup logger
    filepath = f'logs/voting-sim-{datetime.now().strftime("%d-%m-%Y")}.log'
    log = conf_logger(args.log, filepath)

    system: VotingSystem = setup_voting_system(args.voting_system)
    sim = VotingSim(log=log, system=system)
    sim.run()
