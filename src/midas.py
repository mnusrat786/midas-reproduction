"""
MIDAS — NormalCore
==================
Basic MIDAS algorithm (Algorithm 1 in the paper).
Tracks edge counts only. No node counts, no temporal decay.

Score formula (Definition 1):
    score(u,v,t) = (a_uv - s_uv/t)^2 * t^2 / (s_uv * (t-1))

Where:
    a_uv = edges (u→v) in current time tick  [current CMS]
    s_uv = edges (u→v) across all time ticks [total CMS]
    t    = current timestamp
"""

import numpy as np
from .count_min_sketch import CountMinSketch


def compute_score(a: float, s: float, t: int) -> float:
    """
    Chi-squared anomaly score from paper Definition 1.

    Parameters
    ----------
    a : float  — current tick count for this edge
    s : float  — total count for this edge
    t : int    — current timestamp
    """
    if s == 0 or t <= 1:
        return 0.0
    return (a - s / t) ** 2 * t * t / (s * (t - 1))


class MIDAS:
    """
    MIDAS NormalCore — edge-level anomaly detection.

    Parameters
    ----------
    num_rows : int   — CMS rows (hash functions). Default: 2.
    num_cols : int   — CMS columns (buckets). Default: 2719 (ν=0.001).
    seed     : int   — random seed for reproducibility.

    Usage
    -----
    >>> detector = MIDAS(num_rows=2, num_cols=2719, seed=42)
    >>> score = detector(src=1, dst=2, timestamp=5)
    """

    def __init__(self, num_rows: int = 2, num_cols: int = 2719, seed: int = None):
        rng = np.random.default_rng(seed)
        self.cur = CountMinSketch(num_rows, num_cols, rng)   # a_uv: current tick
        self.tot = CountMinSketch(num_rows, num_cols, rng)   # s_uv: all ticks
        self.t   = 1

    def __call__(self, src: int, dst: int, timestamp: int) -> float:
        """
        Process one edge and return its anomaly score.

        Parameters
        ----------
        src       : int — source node ID
        dst       : int — destination node ID
        timestamp : int — time tick of this edge
        """
        if timestamp > self.t:
            self.cur.clear()       # reset current-tick counts
            self.t = timestamp

        self.cur.add_edge(src, dst)
        self.tot.add_edge(src, dst)

        a = self.cur.query_edge(src, dst)
        s = self.tot.query_edge(src, dst)
        return compute_score(a, s, self.t)
