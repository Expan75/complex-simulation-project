import os
import sys
import logging
from voting_system import NaivePlurality, PopularMajority, VotingSystem


# constant of what systems are supported currently
SUPPORTED_VOTING_SYSTEMS = {"plurality": NaivePlurality, "majority": PopularMajority}


def setup_voting_system(name: str, params: dict = {}) -> VotingSystem:
    """Helper for setting up the correct voting system"""
    try:
        return SUPPORTED_VOTING_SYSTEMS[name](params=params)
    except KeyError:
        raise KeyError(f"{name=} is not one of {SUPPORTED_VOTING_SYSTEMS.values()}")


def conf_logger(
    loglevel: int,
    filepath: str,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Helper for configuring logger both to stream stdout and file"""

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
