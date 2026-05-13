"""
Count-Min Sketch (CMS)
======================
A probabilistic data structure for approximate frequency counting.
Uses multiple hash functions to maintain counts in constant memory.

Reference: Cormode & Muthukrishnan, 2005.
"""

import numpy as np


class CountMinSketch:
    """
    Count-Min Sketch with `num_rows` hash functions and `num_cols` buckets.

    Parameters
    ----------
    num_rows : int
        Number of hash functions (rows). More rows = lower error probability.
    num_cols : int
        Number of buckets per row. More cols = lower error magnitude.
        Paper uses 2719 cols → approximation error ν = 0.001.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(self, num_rows: int, num_cols: int, rng=None):
        self.r = num_rows
        self.c = num_cols
        if rng is None:
            rng = np.random.default_rng()
        self.p = 2**31 - 1          # Mersenne prime
        self.a = rng.integers(1, self.p, size=num_rows)
        self.b = rng.integers(0, self.p, size=num_rows)
        self.table = np.zeros((num_rows, num_cols), dtype=np.float32)

    def _hash_edge(self, src: int, dst: int) -> np.ndarray:
        """Hash a (src, dst) pair using Cantor pairing."""
        combined = (src + dst) * (src + dst + 1) // 2 + dst
        return ((self.a * combined + self.b) % self.p % self.c).astype(np.int32)

    def _hash_node(self, node: int) -> np.ndarray:
        """Hash a single node ID."""
        return ((self.a * node + self.b) % self.p % self.c).astype(np.int32)

    def add_edge(self, src: int, dst: int, val: float = 1.0):
        """Increment count for edge (src, dst)."""
        idx = self._hash_edge(src, dst)
        for r, c in enumerate(idx):
            self.table[r, c] += val

    def add_node(self, node: int, val: float = 1.0):
        """Increment count for node."""
        idx = self._hash_node(node)
        for r, c in enumerate(idx):
            self.table[r, c] += val

    def query_edge(self, src: int, dst: int) -> float:
        """Return approximate count for edge (src, dst)."""
        idx = self._hash_edge(src, dst)
        return min(self.table[r, c] for r, c in enumerate(idx))

    def query_node(self, node: int) -> float:
        """Return approximate count for node."""
        idx = self._hash_node(node)
        return min(self.table[r, c] for r, c in enumerate(idx))

    def clear(self):
        """Reset all counts to zero (called at each new time tick in MIDAS)."""
        self.table[:] = 0.0

    def multiply_all(self, factor: float):
        """Multiply all counts by factor (temporal decay in MIDAS-R)."""
        self.table *= factor
