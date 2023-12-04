import argparse
from datetime import datetime

import common
from src.vsim import simulation, electorate, voting_system

# setup cli
parser = argparse.ArgumentParser("vsim", description="Voting simulator 0.0.1")
parser.add_argument(
    "--voting-system",
    "-v",
    choices=voting_system.SUPPORTED_VOTING_SYSTEMS.keys(),
    required=True,
)
parser.add_argument("--candidates", "-c", type=int, default=2)
parser.add_argument("--issues", "-i", type=int, default=2)
parser.add_argument("--population", "-p", type=int, default=10_000)
parser.add_argument("--scenario", "-e", choices=electorate.ELECTORATE_SCENARIOS.keys())
parser.add_argument("--seed", "-s", type=int, default=None)
parser.add_argument("--log", "-l", type=str, default="DEBUG", required=False)
parser.add_argument("--debug", "-d", action="store_true", default=False)
parser.add_argument("--output-dir", "-o", type=str, default="")


def main():
    args = parser.parse_args()

    # setup logger
    filepath = f'logs/voting-sim-{datetime.now().strftime("%d-%m-%Y")}.log'
    log = common.conf_logger(args.log, filepath)

    system = voting_system.setup_voting_system(args.voting_system)
    voters = electorate.setup_electorate(args.population, args.issues, args.scenario)

    sim = simulation.VotingSimulator(
        log=log,
        system=system,
        seed=args.seed,
        plot=args.debug,
        electorate=voters,
        scenario=args.scenario,
        n_candidates=args.candidates,
    )
    sim.run()


if __name__ == "__main__":
    main()
