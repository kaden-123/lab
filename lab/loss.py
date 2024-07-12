import random
import math
from typing import List

def rand_probs(n: int) -> List[float]:
    """Generates n > 0 number of random probabilities from a single event

    Parameters
    ----------
    n : int
        Number of probabilities
    
    Returns
    -------
    List[float]
        List of n probabilites that sum up to 1
    """
    assert n > 0, "n should be above 0"
    cuts = sorted(random.random() for _ in range(n-1))
    return [b-a for a, b in zip([0]+cuts, cuts+[1])]

def entropy(prob_list : List[float]) -> float:
    """Calculates total entropy of a list with n number of probabilities
    
    Parameters
    ----------
    prob_list: List[float]
        List of floats with n probabailities
    
    Returns
    -------
    float
        total entropy of the given list of probabilities
    """
    entropy = 0
    for i in prob_list:
        if(i > 0):
            entropy -= i * math.log(i, 2)
    return entropy

def cross_entropy(
        P: List[float], 
        Q: List[float],
) -> float:
    """Calculates cross entropy given true distrubution(P) vs predicted distrubution(Q)

    Parameters
    ----------
    P : List[float]
        List of probablilities that is the actual or true distrubutions of probabilites
    Q : List[float]
        List of probabilities that is the predicted distrubution

    Returns
    -------
    float
        The cross entropy of true vs predicted distributions
    """
    if len(Q) != len(P):
        raise ValueError("Distrubitions must have same # of probabilities")
        
    Q = [max(q, 1e-10) for q in Q]
    return -sum( p * math.log(q, 2) for p, q in zip(P, Q))
    
def kl_divergence(
        P: List[float],
        Q: List[float],
) -> float:
    """Calculates KL divergence given P and Q 

    Parameters
    ----------
    P : List[float]
        True distribution of probabilities
    Q : List[float]
        Predicted Distrubution of probabilities

    Returns
    -------
    float
        KL divergence of P and Q
    """
    if len(Q) != len(P):
        raise ValueError("Distrubitions must have same # of probabilities")
        
    Q = [max(q, 1e-10) for q in Q]
    P = [max(p, 1e-10) for p in P]

    return sum(p * math.log((p/q),2) for p, q in zip(P, Q))