"""
This module contains strategy classes that implement different voting systems

To add a new voting system, subclass the VotingSystem abstract base calss and be sure to implement the 'elect' method. Have a look at one of the current implementations for help!
"""
import operator
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Set
from dataclasses import dataclass


@dataclass
class ElectionResult:
    winners: Set[int]
    cast_votes: dict


class VotingSystem(ABC):
    """Strategy for running simulator, akin to given system of voting"""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        pass


class NaivePlurality(VotingSystem):
    def __init__(self, *args, **kwargs):
        pass

    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)
            preferred_candidate = np.argmin(distance)
            electoral_vote_count[preferred_candidate] += 1

        # plurality only has a single winner
        winner_idx, _ = max(electoral_vote_count.items(), key=operator.itemgetter(1))
        winners: Set[int] = {winner_idx}
        result = ElectionResult(cast_votes=electoral_vote_count, winners=winners)

        return result


class PopularMajority(VotingSystem):
    def __init__(
        self,
        percentage_threshold: float = 50,
        round_knockoffs: int = 1,
        *args,
        **kwargs,
    ):
        self.threshold = percentage_threshold
        self.knockoffs = round_knockoffs

    def elect_rec(
        self, electorate: np.ndarray, candidates: np.ndarray, prior_results=[]
    ) -> List[ElectionResult]:
        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)
            preferred_candidate = np.argmin(distance)
            electoral_vote_count[preferred_candidate] += 1

        # unlike plurality, unless the winner surpasses 50%, knock out k members from the
        # race and re-elect until a clear majority can be crowned victor.
        plurality_winner_idx, _ = max(
            electoral_vote_count.items(), key=operator.itemgetter(1)
        )
        plurality_winner_share = electoral_vote_count[plurality_winner_idx] / voters
        majority_achieved = plurality_winner_share > 0.5

        result = ElectionResult({plurality_winner_idx}, cast_votes=electoral_vote_count)
        results = [*prior_results, result]

        if majority_achieved:
            return results
        else:
            # drop biggest loser and run second stage
            plurality_worst_loser_idx, _ = min(
                electoral_vote_count.items(), key=operator.itemgetter(1)
            )
            culled_candidates = np.delete(
                candidates, [plurality_worst_loser_idx], axis=0
            )
            return self.elect_rec(electorate, culled_candidates, results)

    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        results: List[ElectionResult] = self.elect_rec(electorate, candidates)
        return results[-1]


class ApprovalVoting(VotingSystem):
    def __init__(self, n_approvals_per_voter: int = 2, *args, **kwargs):
        self.n_approvals_per_voter = n_approvals_per_voter

    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        assert self.n_approvals_per_voter <= n_candidates, "more votes than candidates"
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)
            for top_candidate_id in np.argpartition(
                distance, range(self.n_approvals_per_voter)
            )[: self.n_approvals_per_voter]:
                electoral_vote_count[top_candidate_id] += 1

        winner_idx, _ = max(electoral_vote_count.items(), key=operator.itemgetter(1))
        winners: Set[int] = {winner_idx}
        result = ElectionResult(cast_votes=electoral_vote_count, winners=winners)

        return result


class ProportionalRepresentation(VotingSystem):
    """
    Implements a proportional representation system, which is typically found in
    most parliament elections (Sweden etc.).
    """

    def __init__(
        self,
        seats_to_allocate: int = 349,
        min_share_threshold: float = 0.04,
        *args,
        **kwargs,
    ):
        self.seats: int = seats_to_allocate
        self.threshold: float = min_share_threshold

    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:
        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)
            preferred_candidate = np.argmin(distance)
            electoral_vote_count[preferred_candidate] += 1

        votes_below_tresholds = {
            c: v
            for c, v in electoral_vote_count.items()
            if (v / voters) < self.threshold
        }

        remaining_votes = voters - sum(votes_below_tresholds.values())
        allocated_seats = {
            c: round((v / remaining_votes) * self.seats)
            for c, v in electoral_vote_count.items()
            if c not in votes_below_tresholds
        }

        assert (
            sum(allocated_seats.values()) == self.seats
        ), "all seats were not allocated"

        # loosely defined here, but just seen as candidate with highest seat count
        winner = max(allocated_seats, key=lambda c: allocated_seats[c])
        return ElectionResult(winners={winner}, cast_votes=allocated_seats)


# constant of what systems are supported currently
SUPPORTED_VOTING_SYSTEMS = {
    "plurality": NaivePlurality,
    "majority": PopularMajority,
    "approval": ApprovalVoting,
    "proportional": ProportionalRepresentation,
}


def setup_voting_system(name: str, *args, **kwargs) -> VotingSystem:
    """Helper for setting up the correct voting system"""
    return SUPPORTED_VOTING_SYSTEMS[name](*args, **kwargs)


if __name__ == "__main__":
    pass
