import argparse
from datetime import datetime

import common
import simulation


# setup cli
parser = argparse.ArgumentParser("vsim", description="Voting simulator 0.0.1")
parser.add_argument(
    "--voting-system",
    "-v",
    choices=common.SUPPORTED_VOTING_SYSTEMS.keys(),
    required=True,
)
parser.add_argument("--candidates", "-c", type=int, default=2)
parser.add_argument("--population", "-p", type=int, default=10_000)
parser.add_argument("--seed", "-s", type=int, default=None)
parser.add_argument("--log", "-l", type=str, default="DEBUG", required=False)


def main():
    args = parser.parse_args()

    # setup logger
    filepath = f'logs/voting-sim-{datetime.now().strftime("%d-%m-%Y")}.log'
    log = common.conf_logger(args.log, filepath)

    system = common.setup_voting_system(args.voting_system)
    sim = simulation.VotingSimulator(
        log=log,
        system=system,
        seed=args.seed,
        n_candidates=args.candidates,
        population_size=args.population,
    )
    sim.run()


if __name__ == "__main__":
    main()
