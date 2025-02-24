import functools
from typing import Iterable, Literal

import numpy as np
from scipy import integrate
from scipy.stats import rv_continuous


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


def log_score(
    x: float,
    dist: rv_continuous,
    params: tuple[float, ...],
    support: tuple[float, float],
) -> float:
    if support[0] <= x <= support[1]:
        pdf = dist.pdf(x, *params)
    else:
        pdf = 0
    pdf = max(pdf, 1e-5)
    return -np.log(pdf)


@functools.cache
def _crps_cdf(
    dist: rv_continuous,
    params: tuple[float, ...],
    x_range: tuple[float, float],
    n_pts: int = 1000,
):
    xs = np.linspace(start=x_range[0], stop=x_range[1], num=n_pts)
    return xs, dist.cdf(xs, *params)


def crps(
    x: float,
    dist: rv_continuous,
    params: tuple[float, ...],
    support: tuple[float, float],
    approx: bool = True,
) -> float:
    if approx:
        xs, cdf = _crps_cdf(dist, params, support)
        if x < support[0]:
            return integrate.trapezoid((1 - cdf) ** 2, xs)
        elif x > support[1]:
            return integrate.trapezoid(cdf**2, xs)
        assert xs[0] <= x <= xs[-1], (x, xs[0], xs[-1])
        # Find where x sits in the xs array
        idx = np.searchsorted(xs, x)
        # Insert the x value and its corresponding cdf value into the array
        cdf = np.insert(cdf, idx, np.interp(x, xs, cdf))
        xs = np.insert(xs, idx, x)
        # Compute score
        score = integrate.trapezoid(cdf[: idx + 1] ** 2, xs[: idx + 1])
        score += integrate.trapezoid((1 - cdf[idx:]) ** 2, xs[idx:])

    else:
        score = (
            integrate.quad(lambda x: dist.cdf(x, *params) ** 2, a=support[0], b=x)[0]
            + integrate.quad(
                lambda x: (1 - dist.cdf(x, *params)) ** 2, a=x, b=support[1]
            )[0]
        )
    return score


def mean_score(
    xs: np.ndarray,
    counts: np.ndarray,
    dist: rv_continuous,
    params: tuple[float, ...],
    scorer: Literal["log", "crps"],
) -> float:
    assert len(xs) == len(counts)
    func = {"log": log_score, "crps": crps}[scorer]
    support = (min(xs) - 20, max(xs) + 20)
    return (
        sum(
            func(x, dist, params, support) * count
            for x, count in zip(xs, counts)
            if count != 0
        )
        / counts.sum()
    )
