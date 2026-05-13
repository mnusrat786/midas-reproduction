"""
SedanSpot — Baseline
=====================
Python reimplementation of SedanSpot (Eswaran & Faloutsos, ICDM 2018).
https://github.com/dhivyaeswaran/sedanspot

Algorithm:
  1. Maintain a weighted reservoir sample of edges (size=500)
  2. Score each edge = increase in personalized PageRank visit fraction
     from src to dst, estimated via short geometric random walks
  3. Anomaly = edge that increases PPR score (bridge-like / sparse edges)

Paper parameters: sample_size=500, num_walks=50, restart_prob=0.15
"""

import math
import heapq
import random
import time
import numpy as np
from collections import defaultdict


class LazyAliasTable:
    """Weighted adjacency list for one source node — supports O(1) sampling."""

    def __init__(self):
        self.weights = {}
        self.total   = 0.0

    def increment(self, dst, wt):
        self.weights[dst] = self.weights.get(dst, 0.0) + wt
        self.total += wt

    def decrement(self, dst, wt):
        self.weights[dst] = self.weights.get(dst, 0.0) - wt
        self.total -= wt
        if self.weights.get(dst, 0.0) <= 0:
            self.weights.pop(dst, None)

    def sample_neighbor(self):
        if not self.weights:
            return None
        r = random.random() * self.total
        cumsum = 0.0
        for dst, w in self.weights.items():
            cumsum += w
            if r <= cumsum:
                return dst
        return next(iter(self.weights))


class SedanSpot:
    """
    SedanSpot — streaming edge anomaly detection via personalized PageRank.

    Parameters
    ----------
    sample_size  : int   — reservoir size. Default: 500.
    num_walks    : int   — random walks per edge. Default: 50.
    restart_prob : float — geometric walk length parameter. Default: 0.15.
    seed         : int   — random seed.
    """

    def __init__(self, sample_size: int = 500, num_walks: int = 50,
                 restart_prob: float = 0.15, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.sample_size  = sample_size
        self.num_walks    = num_walks
        self.eps          = restart_prob

        self.heap         = []
        self.heap_counter = 0
        self.entries_seen = 0

        self.adj       = defaultdict(LazyAliasTable)
        self.src_count = defaultdict(int)
        self.dst_count = defaultdict(int)

        self._cur_time  = None
        self._prev_time = -1
        self._batch     = []
        self._scores    = []

    def _sample(self, src, dst, wt, sampling_weight):
        self.entries_seen += 1
        self.heap_counter += 1
        priority = math.log10(random.random() + 1e-300) / max(sampling_weight, 1e-10)

        if self.entries_seen <= self.sample_size:
            heapq.heappush(self.heap, (priority, self.heap_counter, src, dst, wt))
            return None, (src, dst, wt)
        else:
            if self.heap and self.heap[0][0] < priority:
                removed = self.heap[0]
                heapq.heapreplace(self.heap, (priority, self.heap_counter, src, dst, wt))
                return (removed[2], removed[3], removed[4]), (src, dst, wt)
            return None, None

    def _add_edge(self, src, dst, wt=1.0):
        self.adj[src].increment(dst, wt)
        self.src_count[src] += 1
        self.dst_count[dst] += 1

    def _remove_edge(self, src, dst, wt=1.0):
        self.adj[src].decrement(dst, wt)
        self.src_count[src] -= 1
        if self.src_count[src] <= 0:
            self.src_count.pop(src, None)
            self.adj.pop(src, None)
        self.dst_count[dst] -= 1
        if self.dst_count[dst] <= 0:
            self.dst_count.pop(dst, None)

    def _visit_fraction(self, src, dst, extra_edge=None):
        """Estimate PPR visit fraction from src to dst via random walks."""
        num_visits = 0
        num_steps  = 0
        for _ in range(self.num_walks):
            walk_len = np.random.geometric(self.eps)
            cur = src
            num_steps += walk_len
            for _ in range(walk_len):
                if cur == dst:
                    num_visits += 1
                lat = self.adj.get(cur)
                if extra_edge and cur == extra_edge[0]:
                    base_wt  = lat.total if lat else 0.0
                    total_wt = base_wt + 1.0
                    if total_wt <= 0:
                        break
                    r = random.random() * total_wt
                    if r > base_wt:
                        cur = extra_edge[1]
                    elif lat:
                        nxt = lat.sample_neighbor()
                        cur = nxt if nxt else cur
                    else:
                        break
                elif lat:
                    nxt = lat.sample_neighbor()
                    cur = nxt if nxt else cur
                else:
                    break
        return num_visits / num_steps if num_steps > 0 else 0.0

    def _score_edge(self, src, dst):
        if len(self.heap) < self.sample_size:
            return 0.0
        before = self._visit_fraction(src, dst)
        after  = self._visit_fraction(src, dst, extra_edge=(src, dst))
        return max(0.0, after - before)

    def _flush_batch(self):
        if not self._batch:
            return
        n_b = len(self._batch)
        dt  = self._cur_time - self._prev_time if self._prev_time >= 0 else 1
        sampling_weight = max(dt / n_b, 1e-10)

        for s, d, _ in self._batch:
            self._scores.append(self._score_edge(s, d))

        for s, d, w in self._batch:
            removed, added = self._sample(s, d, w, sampling_weight)
            if removed:
                self._remove_edge(*removed)
            if added:
                self._add_edge(*added)

        self._prev_time = self._cur_time
        self._batch = []

    def __call__(self, src: int, dst: int, timestamp: int) -> float:
        src_s, dst_s = str(src), str(dst)
        if self._cur_time is None:
            self._cur_time = timestamp
        if timestamp != self._cur_time:
            self._flush_batch()
            self._cur_time = timestamp
        self._batch.append((src_s, dst_s, 1.0))
        return 0.0

    def finalize(self):
        """Flush the last batch. Must be called after processing all edges."""
        self._flush_batch()

    def get_scores(self):
        return np.array(self._scores, dtype=np.float32)


def run_sedanspot(src_arr, dst_arr, ts_arr,
                  sample_size: int = 500, num_walks: int = 50,
                  restart_prob: float = 0.15, seed: int = 42):
    """
    Run SedanSpot on a full edge stream.

    Returns
    -------
    scores  : np.ndarray of shape (n,)
    elapsed : float — total seconds
    """
    n   = len(src_arr)
    det = SedanSpot(sample_size=sample_size, num_walks=num_walks,
                    restart_prob=restart_prob, seed=seed)
    t0  = time.perf_counter()
    for i in range(n):
        det(int(src_arr[i]), int(dst_arr[i]), int(ts_arr[i]))
        if i % 500_000 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            print(f"    SedanSpot: {i:,}/{n:,} ({i/n*100:.0f}%) "
                  f"elapsed={elapsed:.0f}s  ETA={elapsed/i*(n-i):.0f}s")
    det.finalize()
    elapsed = time.perf_counter() - t0
    scores  = det.get_scores()
    if len(scores) < n:
        scores = np.pad(scores, (0, n - len(scores)))
    return scores[:n], elapsed
