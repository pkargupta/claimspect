from tqdm import tqdm
import re
import numpy as np
from collections import deque, Counter
import json

class Paper:
    def __init__(self, id, title, abstract, text):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = text # f"title : {title}; abstract: {abstract}"
        self.split_text = self.text.split(" ")
        self.vocabulary = dict(Counter(self.split_text))

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.id == other.id)

    def __repr__(self) -> str:
        return self.text

class Aspect:
    def __init__(self, label, keywords):
        self.aspect = label
        self.seeds = keywords
        self.hierarchy = None

class Perspective:
    def __init__(self) -> None:
        self.

class Hierarchy:
    def __init__(self, aspect) -> None:
        self.aspect = aspect
        self.root = 