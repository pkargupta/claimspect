"""
Abstract class for segment filter.
The input of the function should at least contain the following fields:
- 'claim': the nuanced claim
- 'list_of_segments': a list of segments
- other fields that are needed for the specific filter
The output of the function should be a subset of the input list_of_segments that is relevant to the claim.
- 'relevant_segments': a list of relevant segments
"""

from abc import ABC, abstractmethod

class AbstractFilter(ABC):
    @abstractmethod
    def filter(self, 
               claim: str, 
               list_of_segments: list[str], 
               **kwargs) -> list[str]:
        pass
    
