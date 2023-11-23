import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

VOTING_SYSTEMS = {"plurality-with-runoff"}


parser = argparse.ArgumentParser("voting-sim")
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
    cast_votes: Dict[str, str]


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
        pass


class VotingSim:
    """Singleton for running the simulation"""

    def __init__(
        self,
        voters: int,
        questions: int,
        candidates: int,
        log: logging.Logger,
        system: VotingSystem,
        seed: Optional[int] = None,
    ):
        np.random.seed(seed)  # None means random without seed
        self.log = log
        self.t_step_size = 1
        self.voting_system: VotingSystem = system

    def generate_electorate(self) -> np.ndarray:
        return np.random.rand(self.voters, self.questions)

    def generate_candidates(self) -> np.ndarray:
        return np.random.rand(self.candidates, self.questions)

    def compute_outcome_metrics(self, results: ElectionResult) -> pd.DataFrame:
        return pd.DataFrame()

    def run_election(self):
        voters = self.generate_electorate()
        issues = self.generate_candidates()
        results = self.voting_system.elect(self.voters, self.issues)
        metrics = self.compute_outcome_metrics(results)

    def run(self):
        """Runs the simulation"""
        self.log.debug("running voting sim")


def setup_voting_system(choice: str, parameters: dict) -> VotingSystem:
    return NaivePlurality({})


if __name__ == "__main__":
    args = parser.parse_args()

    # setup logger
    filepath = f'logs/voting-sim-{datetime.now().strftime("%d-%m-%Y")}.log'
    log = conf_logger(args.log, filepath)

    system: VotingSystem = setup_voting_system(args.voting_system)
    sim = VotingSim(log=log, system=system_to_simulate)
