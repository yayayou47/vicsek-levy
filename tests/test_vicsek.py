"""Light tests on the dynamics."""
import numpy as np

from vicsek import Vicsek, VicsekParams


def test_periodic_boundaries():
    p = VicsekParams(N=200, L=10.0, eta=0.0, alpha=2.0, seed=0)
    sim = Vicsek(p)
    for _ in range(20):
        sim.step()
    assert sim.x.min() >= 0.0 and sim.x.max() < p.L
    assert sim.y.min() >= 0.0 and sim.y.max() < p.L


def test_zero_noise_increases_order():
    p = VicsekParams(N=400, L=15.0, v0=0.05, eta=0.0, alpha=2.0, seed=42)
    sim = Vicsek(p)
    phi0 = sim.polarisation()
    for _ in range(500):
        sim.step()
    assert sim.polarisation() > max(0.5, phi0)


def test_high_noise_disorders():
    p = VicsekParams(N=400, L=15.0, v0=0.05, eta=3.0, alpha=2.0, seed=42)
    sim = Vicsek(p)
    for _ in range(300):
        sim.step()
    assert sim.polarisation() < 0.3
