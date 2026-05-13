"""
MIDAS-R — RelationalCore
=========================
Extended MIDAS with spatial and temporal relations (Algorithm 2 in the paper).

Improvements over MIDAS:
  1. Spatial: tracks node-level counts (src and dst separately)
             → catches bursts from one IP to many destinations
  2. Temporal: instead of resetting current counts each tick,
               multiply by decay factor alpha ∈ (0,1)
             → recent edges count more than older ones

Final score = max(edge_score, src_score, dst_score)
"""

import numpy as np
from .count_min_sketch import CountMinSketch
from .midas import compute_score


class MIDAS_R:
    """
    MIDAS-R RelationalCore — edge + node level anomaly detection with decay.

    Parameters
    ----------
    num_rows : int   — CMS rows. Default: 2.
    num_cols : int   — CMS columns. Default: 2719.
    alpha    : float — temporal decay factor. Default: 0.5 (paper setting).
    seed     : int   — random seed.

    Usage
    -----
    >>> detector = MIDAS_R(num_rows=2, num_cols=2719, alpha=0.5, seed=42)
    >>> score = detector(src=1, dst=2, timestamp=5)
    """

    def __init__(self, num_rows: int = 2, num_cols: int = 2719,
                 alpha: float = 0.5, seed: int = None):
        rng = np.random.default_rng(seed)
        self.alpha = alpha

        # Edge-level CMS (same as MIDAS)
        self.cur_edge = CountMinSketch(num_rows, num_cols, rng)
        self.tot_edge = CountMinSketch(num_rows, num_cols, rng)

        # Source node-level CMS
        self.cur_src  = CountMinSketch(num_rows, num_cols, rng)
        self.tot_src  = CountMinSketch(num_rows, num_cols, rng)

        # Destination node-level CMS
        self.cur_dst  = CountMinSketch(num_rows, num_cols, rng)
        self.tot_dst  = CountMinSketch(num_rows, num_cols, rng)

        self.t = 1

    def __call__(self, src: int, dst: int, timestamp: int) -> float:
        """
        Process one edge and return its anomaly score.

        Parameters
        ----------
        src       : int — source node ID
        dst       : int — destination node ID
        timestamp : int — time tick
        """
        if timestamp > self.t:
            # Temporal decay: multiply current counts by alpha (not reset)
            self.cur_edge.multiply_all(self.alpha)
            self.cur_src.multiply_all(self.alpha)
            self.cur_dst.multiply_all(self.alpha)
            # Merge decayed current into total
            self.tot_edge.table += self.cur_edge.table
            self.tot_src.table  += self.cur_src.table
            self.tot_dst.table  += self.cur_dst.table
            self.t = timestamp

        # Update counts
        self.cur_edge.add_edge(src, dst)
        self.cur_src.add_node(src)
        self.cur_dst.add_node(dst)

        # Query counts
        a_e = self.cur_edge.query_edge(src, dst)
        s_e = self.tot_edge.query_edge(src, dst)
        a_s = self.cur_src.query_node(src)
        s_s = self.tot_src.query_node(src)
        a_d = self.cur_dst.query_node(dst)
        s_d = self.tot_dst.query_node(dst)

        # Final score = max of edge, source, destination scores
        return max(
            compute_score(a_e, s_e, self.t),
            compute_score(a_s, s_s, self.t),
            compute_score(a_d, s_d, self.t),
        )
