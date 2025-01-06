from typing import Iterable

import numpy as np
from scipy.integrate import quad


def histogram_to_data(bins: Iterable[float], counts: Iterable[int]) -> np.ndarray:
    return np.hstack(
        [np.ones((count,), dtype=int) * value for value, count in zip(bins, counts)]
    )


def moment_0(shape: float):
    # https://en.wikipedia.org/wiki/Skew_normal_distribution#Definition
    delta = shape / np.sqrt(1 + shape * shape)
    two_on_pi = 2 / np.pi
    mode = np.sqrt(two_on_pi) * delta
    mode -= (
        (1 - np.pi / 4)
        * np.power(np.sqrt(two_on_pi) * delta, 3)
        / (1 - two_on_pi * delta * delta)
    )
    if shape != 0:
        mode -= (np.sign(shape) / 2) * np.exp(-2 * np.pi / abs(shape))
    return mode


def skewnorm_mode(shape: float, loc: float, scale: float) -> float:
    return loc + scale * moment_0(shape)


def rescale_peak(hist: np.ndarray, max_peak_ratio: float) -> np.ndarray:
    hist = hist.copy()
    idxmax = hist.argmax()
    assigned_proportion = hist[idxmax] / hist.sum()
    scale = (
        (hist[idxmax] - assigned_proportion * hist[idxmax])
        * max_peak_ratio
        / ((hist[idxmax] - max_peak_ratio * hist[idxmax]) * assigned_proportion)
    )
    hist[idxmax] *= min(scale, 1.0)
    return hist


def log_score(data, dist, *params):
    unique, counts = np.unique(data, return_counts=True)
    pdf = dist.pdf(unique, *params)
    return (-np.log(np.maximum(pdf, 1e-5)) * counts).sum() / counts.sum()


def crps(data, dist, *params):
    total = 0
    seen = set()
    for g in data:
        if g in seen:
            continue
        count = (data == g).sum()
        total += count * quad(lambda x: dist.cdf(x, *params) ** 2, a=-np.inf, b=g)[0]
        total += (
            count * quad(lambda x: (1 - dist.cdf(x, *params)) ** 2, a=g, b=np.inf)[0]
        )
        seen.add(g)
    return total / len(data)
