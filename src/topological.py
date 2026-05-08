"""
Topological variant of the zonal Vicsek-Couzin model.

Repulsion is metric (d < R_r), alignment is over the k nearest
neighbours under periodic boundaries, restricted to the vision cone.
We query k_max >= k nearest neighbours via cKDTree, then a numba
kernel filters them per-particle to apply repulsion-then-alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from noise import stable_rvs

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@dataclass
class TopoParams:
    N: int = 500
    L: float = 15.0
    v0: float = 0.05
    R_r: float = 0.5
    k: int = 6
    beta: float = 30.0
    eta: float = 0.3
    alpha: float = 2.0
    seed: int = 0


@njit(cache=True, parallel=True)
def _topo_update(x, y, theta, knn, L, R_r, k_use, blind_cos):
    """For each particle, sweep its (k_max) nearest neighbours; identify
    repulsion neighbours (d < R_r) and alignment neighbours (the next
    k_use that are not in the repulsion zone), within the vision cone.
    """
    N = x.shape[0]
    new_theta = np.empty(N)
    halfL = 0.5 * L
    Rr2 = R_r * R_r
    k_max = knn.shape[1]
    for i in prange(N):
        hcx = np.cos(theta[i])
        hcy = np.sin(theta[i])
        rep_x = 0.0
        rep_y = 0.0
        n_rep = 0
        ali_s = 0.0
        ali_c = 0.0
        n_ali = 0
        for m in range(k_max):
            j = knn[i, m]
            if j == i or j < 0:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx > halfL:
                dx -= L
            elif dx < -halfL:
                dx += L
            if dy > halfL:
                dy -= L
            elif dy < -halfL:
                dy += L
            d2 = dx * dx + dy * dy
            if d2 <= 0.0:
                continue
            d = np.sqrt(d2)
            ux = dx / d
            uy = dy / d
            if hcx * ux + hcy * uy < blind_cos:
                continue
            if d2 < Rr2:
                rep_x -= ux
                rep_y -= uy
                n_rep += 1
            elif n_ali < k_use:
                ali_s += np.sin(theta[j])
                ali_c += np.cos(theta[j])
                n_ali += 1
        if n_rep > 0:
            new_theta[i] = np.arctan2(rep_y, rep_x)
        elif n_ali > 0:
            new_theta[i] = np.arctan2(ali_s, ali_c)
        else:
            new_theta[i] = theta[i]
    return new_theta


class TopoVicsek:
    def __init__(self, p: TopoParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.x = self.rng.uniform(0, p.L, size=p.N)
        self.y = self.rng.uniform(0, p.L, size=p.N)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=p.N)
        half = np.deg2rad(0.5 * p.beta)
        self.blind_cos = -np.cos(half)
        # Query k_max nearest -- enough to absorb a few repulsion
        # neighbours plus the topological alignment quota.
        self.k_max = max(p.k + 4, 12)
        self.t = 0

    def step(self) -> None:
        p = self.p
        L = p.L
        pts = np.column_stack([self.x % L, self.y % L])
        tree = cKDTree(pts, boxsize=L)
        _, knn = tree.query(pts, k=self.k_max + 1)
        # query returns shape (N, k_max+1); the first column is self.
        knn = knn.astype(np.int64)
        target = _topo_update(
            self.x, self.y, self.theta, knn,
            L, p.R_r, p.k, self.blind_cos,
        )
        xi = stable_rvs(p.alpha, p.eta, p.N, self.rng)
        self.theta = (target + xi + np.pi) % (2 * np.pi) - np.pi
        self.x = (self.x + p.v0 * np.cos(self.theta)) % L
        self.y = (self.y + p.v0 * np.sin(self.theta)) % L
        self.t += 1

    def polarisation(self) -> float:
        return float(np.hypot(np.mean(np.cos(self.theta)),
                              np.mean(np.sin(self.theta))))
