"""
This module contains strategy classes that implement different voting systems

To add a new voting system, subclass the VotingSystem abstract base calss and be sure to implement the 'elect' method. Have a look at one of the current implementations for help!
"""
import operator
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Set
from dataclasses import dataclass


__all__ = ["VotingSystem", "NaivePlurality", "PopularMajority", "ElectionResult"]


@dataclass
class ElectionResult:
    winners: Set[int]
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
        super().__init__(params)

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
    def __init__(self, params: dict):
        super().__init__(params)

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


class ApprovalVoting(VotingSystem):  # får rösta på hur många partier som helst. Den med flest röster vinner, 
    # de k närmaste kandidaterna röstar varje electorate på då nrVotes
    def __init__(self, params: dict):
        self.params = params
        #self.nrVotes = params.get("nrVotes", 3) # är rätt? 3 om ej valt
        #dict 
    def elect(self, electorate: np.ndarray, candidates: np.ndarray) -> ElectionResult:

        voters, _ = electorate.shape
        n_candidates, _ = candidates.shape
        electoral_vote_count = {i: 0 for i in range(n_candidates)}

        #self.nrVotes = min(self.nrVotes, n_candidates - 1) # så att inte mer än antalet kandidater och -1 för index 
        assert self.params["nrVotes"] > n_candidates, "more votes than candidates"
        for voter_i in range(voters):
            distance = np.linalg.norm(candidates - electorate[voter_i, :], axis=1)

            top_candidate_idx = np.argpartition(distance, self.params["nrVotes"])[:self.params["nrVotes"]] # ger array med indices för k närmaste kandidaterna
            
            for i_candidate in top_candidate_idx: # går igenom alla indices för 
                electoral_vote_count[i_candidate] += 1

        winner_idx, _ = max(electoral_vote_count.items(), key=operator.itemgetter(1))
        winners: Set[int] = {winner_idx}
        result = ElectionResult(cast_votes=electoral_vote_count, winners=winners)

        return result
    

# constant of what systems are supported currently
SUPPORTED_VOTING_SYSTEMS = {"plurality": NaivePlurality, "majority": PopularMajority, "approval":ApprovalVoting}

def setup_voting_system(name: str, params: dict = {}) -> VotingSystem:
    """Helper for setting up the correct voting system"""
    try:
        return SUPPORTED_VOTING_SYSTEMS[name](params=params)
    except KeyError:
        raise KeyError(f"{name=} is not one of {SUPPORTED_VOTING_SYSTEMS.values()}")
