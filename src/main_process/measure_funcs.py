import os
import sys
from typing import Optional, Tuple, List
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cassis

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def verbs_per_sentence(cas: cassis.Cas) -> List[int]:
    """
    Calculates how many verbs are in a sentence.
    :param cas:
    :return:
    """
    pass


def token_per_sentence(cas: cassis.Cas) -> List[int]:
    """
    Calculates how many tokens are in a sentence.
    :param cas:
    :return:
    """
    pass


def mdd_of_sent(cas: cassis.Cas) -> List[float]:
    """
    For a Pair of dependent and governor the distance of their position
    in a sentence is calculated.
    The mdd ist the mean for all those pairs in a sentence.
    :param cas:
    :return:
    """


def altmann(cas: cassis.Cas) -> List[float]:
    """
    MAL states that with increasing size of a linguistic construct, the size of its parts shrinks.
    ??? Calculation is the mean of the size of its parts ???
    :param cas:
    :return:
    """