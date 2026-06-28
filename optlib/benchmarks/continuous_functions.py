"""
optlib/benchmarks/continuous_functions.py
Standard test functions for continuous optimisation.

All functions are **minimisation** targets.
Each entry in REGISTRY is:
    name -> {"func": callable, "bounds": [(lo,hi)]*n, "global_min": float,
             "global_min_x": ndarray | None, "description": str}
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Individual functions ───────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def rosenbrock(x: np.ndarray) -> float:
    xi, xi1 = x[:-1], x[1:]
    return float(np.sum(100.0 * (xi1 - xi**2)**2 + (1.0 - xi)**2))


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2,
           c: float = 2.0 * math.pi) -> float:
    n = len(x)
    s1 = np.sqrt(np.dot(x, x) / n)
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * math.exp(-b * s1) - math.exp(s2) + a + math.e)


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x)))


def schwefel(x: np.ndarray) -> float:
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def griewank(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(np.sum(x**2) / 4000.0
                 - np.prod(np.cos(x / np.sqrt(idx)))
                 + 1.0)


def levy(x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    t1 = math.sin(math.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(math.pi * w[:-1] + 1.0)**2))
    t3 = (w[-1] - 1.0)**2 * (1.0 + math.sin(2.0 * math.pi * w[-1])**2)
    return float(t1 + t2 + t3)


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    i = np.arange(1, len(x) + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(i * x**2 / math.pi) ** (2 * m)))


def dixon_price(x: np.ndarray) -> float:
    t1 = (x[0] - 1.0)**2
    i  = np.arange(2, len(x) + 1, dtype=float)
    t2 = np.sum(i * (2.0 * x[1:]**2 - x[:-1])**2)
    return float(t1 + t2)


def zakharov(x: np.ndarray) -> float:
    i   = np.arange(1, len(x) + 1, dtype=float)
    t1  = np.dot(x, x)
    sum2 = np.dot(0.5 * i, x)
    return float(t1 + sum2**2 + sum2**4)


def alpine1(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def styblinski_tang(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


# 2-D only ──────────────────────────────────────────────────────────────────

def himmelblau(xy: np.ndarray) -> float:
    x, y = xy[0], xy[1]
    return float((x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2)


def beale(xy: np.ndarray) -> float:
    x, y = xy[0], xy[1]
    return float((1.5 - x + x*y)**2
                 + (2.25 - x + x*y**2)**2
                 + (2.625 - x + x*y**3)**2)


def eggholder(xy: np.ndarray) -> float:
    x, y = xy[0], xy[1]
    return float(-(y + 47.0) * math.sin(math.sqrt(abs(x/2.0 + y + 47.0)))
                 - x * math.sin(math.sqrt(abs(x - (y + 47.0)))))


def bukin6(xy: np.ndarray) -> float:
    x, y = xy[0], xy[1]
    return float(100.0 * math.sqrt(abs(y - 0.01 * x**2)) + 0.01 * abs(x + 10.0))


# ── Registry ───────────────────────────────────────────────────────────────

def make_registry(n: int = 10) -> Dict:
    one = np.ones(n)
    return {
        "sphere": {
            "func": sphere,
            "bounds": [(-5.12, 5.12)] * n,
            "global_min": 0.0,
            "global_min_x": np.zeros(n),
            "description": "Sphere – convex, unimodal baseline",
        },
        "rosenbrock": {
            "func": rosenbrock,
            "bounds": [(-2.048, 2.048)] * n,
            "global_min": 0.0,
            "global_min_x": one.copy(),
            "description": "Rosenbrock – narrow curved valley, hard to converge",
        },
        "ackley": {
            "func": ackley,
            "bounds": [(-32.768, 32.768)] * n,
            "global_min": 0.0,
            "global_min_x": np.zeros(n),
            "description": "Ackley – exponential landscape, many local minima",
        },
        "rastrigin": {
            "func": rastrigin,
            "bounds": [(-5.12, 5.12)] * n,
            "global_min": 0.0,
            "global_min_x": np.zeros(n),
            "description": "Rastrigin – highly multimodal (10n local minima)",
        },
        "schwefel": {
            "func": schwefel,
            "bounds": [(-500.0, 500.0)] * n,
            "global_min": 0.0,
            "global_min_x": np.full(n, 420.9687),
            "description": "Schwefel – deceptive; best solution far from 2nd best",
        },
        "griewank": {
            "func": griewank,
            "bounds": [(-600.0, 600.0)] * n,
            "global_min": 0.0,
            "global_min_x": np.zeros(n),
            "description": "Griewank – widespread local minima, product term",
        },
        "levy": {
            "func": levy,
            "bounds": [(-10.0, 10.0)] * n,
            "global_min": 0.0,
            "global_min_x": one.copy(),
            "description": "Levy – irregular local minima near the boundary",
        },
        "michalewicz": {
            "func": michalewicz,
            "bounds": [(0.0, math.pi)] * n,
            "global_min": None,          # depends on n
            "global_min_x": None,
            "description": "Michalewicz – steep ridges, long flat valleys",
        },
        "dixon_price": {
            "func": dixon_price,
            "bounds": [(-10.0, 10.0)] * n,
            "global_min": 0.0,
            "global_min_x": None,        # analytic but dimension-dependent
            "description": "Dixon-Price – unimodal but ill-conditioned",
        },
        "zakharov": {
            "func": zakharov,
            "bounds": [(-5.0, 10.0)] * n,
            "global_min": 0.0,
            "global_min_x": np.zeros(n),
            "description": "Zakharov – flat near origin, then grows fast",
        },
        "styblinski_tang": {
            "func": styblinski_tang,
            "bounds": [(-5.0, 5.0)] * n,
            "global_min": -39.16599 * n,
            "global_min_x": np.full(n, -2.903534),
            "description": "Styblinski-Tang – multiple local minima per dim",
        },
        "himmelblau_2d": {
            "func": himmelblau,
            "bounds": [(-5.0, 5.0)] * 2,
            "global_min": 0.0,
            "global_min_x": np.array([3.0, 2.0]),   # one of 4 global minima
            "description": "Himmelblau – 4 global minima (2-D only)",
        },
        "beale_2d": {
            "func": beale,
            "bounds": [(-4.5, 4.5)] * 2,
            "global_min": 0.0,
            "global_min_x": np.array([3.0, 0.5]),
            "description": "Beale – flat near boundary, steep centre (2-D only)",
        },
        "eggholder_2d": {
            "func": eggholder,
            "bounds": [(-512.0, 512.0)] * 2,
            "global_min": -959.6407,
            "global_min_x": np.array([512.0, 404.2319]),
            "description": "Eggholder – very many local minima (2-D only)",
        },
    }


REGISTRY = make_registry(n=10)
REGISTRY_2D = {k: v for k, v in REGISTRY.items() if "2d" in k}
REGISTRY_ND = {k: v for k, v in REGISTRY.items() if "2d" not in k}
