import os
import sys
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional


VOTING_SYSTEMS = {"plurality-with-runoff"}


parser = argparse.ArgumentParser("voting-sim")
parser.add_argument("--voting-system", "-v", choices=VOTING_SYSTEMS)
parser.add_argument("--timesteps", "-t", type=int, default=0)
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


class VotingSim:
    def __init__(self, log: logging.Logger):
        self.log = log
        self.t = 0
        self.t_step_size = 1

    def tick(self):
        self.t += self.t_step_size

    def run(self, timesteps: int):
        """Runs the simulation"""
        self.log.debug("running voting sim")
        for _ in tqdm(range(1, timesteps + 1)):
            self.tick()


if __name__ == "__main__":
    args = parser.parse_args()

    # setup logger
    filepath = f'logs/voting-sim-{datetime.now().strftime("%d-%m-%Y")}.log'
    log = conf_logger(args.log, filepath)
    sim = VotingSim(log=log)
    sim.run(10)
