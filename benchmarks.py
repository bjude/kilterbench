from typing import Iterable
from functools import cache

import numpy as np
import pandas as pd
from scipy.optimize import newton, minimize, basinhopping
from scipy.integrate import quad
from scipy.stats import skewnorm
import sqlite3

from kilter_api import KilterAPI


@cache
def grade_histogram(session: KilterAPI, climb_id: str, angle: int) -> np.ndarray:
    stats = session.get_climb_stats(climb_id, angle)
    grades = np.zeros(39, dtype=int)
    for s in stats["difficulty"]:
        grades[s["difficulty"] - 1] = s["count"]
    return grades


def histogram_to_data(bins: Iterable[float], counts: Iterable[int]) -> np.ndarray:
    return np.hstack(
        [np.ones((count,), dtype=int) * value for value, count in zip(bins, counts)]
    )


class WrappedDataException(RuntimeError):
    pass


def moment_0(shape: float):
    # https://en.wikipedia.org/wiki/Skew_normal_distribution#Definition
    delta = shape / np.sqrt(1 + shape * shape)
    two_on_pi = 2 / np.pi
    mode = np.sqrt(two_on_pi) * delta
    mode -= (
        (1 - np.pi / 4)
        * np.pow(np.sqrt(two_on_pi) * delta, 3)
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
    pdf = dist.pdf(data, *params)
    return -np.log(np.maximum(pdf, 1e-5)).mean()


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


def fit_grade_curve(grade_histogram: np.ndarray) -> tuple[float, float, float]:
    # grade_histogram = grade_histogram.copy()

    def mode_delta(x: float, data: np.ndarray, target_mode: float):
        params = skewnorm.fit(data, floc=x)
        return skewnorm_mode(*params) - target_mode

    def opt_func(params, data, target_mode, weight):
        params[2] = max(params[2], 1e-8)
        shape, loc, scale = params
        delta = mode_delta(loc, data, target_mode)
        # score = crps(data, skewnorm, *params)
        score = log_score(data, skewnorm, *params)
        return delta * delta + weight * score  # + np.square(shape).sum() * 0.0001

    grades = np.arange(1, 40)
    idxmax = grade_histogram.argmax()
    assigned_grade: int = grades[idxmax]

    target_assigned_proportion = 0.5
    grade_histogram = rescale_peak(grade_histogram, target_assigned_proportion)

    grade_data = histogram_to_data(grades, grade_histogram)
    params = skewnorm.fit(grade_data)
    # margin = 1.0
    # if abs(delta := (skewnorm_mode(*params) - assigned_grade)) > margin:
    #    target_grade = assigned_grade + margin * (1 if delta > 0 else -1)
    #    floc, res = newton(
    #        mode_delta,
    #        assigned_grade,
    #        args=(grade_data, target_grade),
    #        disp=False,
    #        full_output=True,
    #        tol=1e-5,
    #    )
    #    params = skewnorm.fit(grade_data, floc=floc)
    #    if not res.converged:
    #        raise WrappedDataException(
    #            params, res, mode_delta(floc, grade_data, assigned_grade)
    #        )
    weight = 10
    # res = minimize(opt_func, params, (grade_data, assigned_grade, weight))
    res = minimize(
        opt_func, (0, assigned_grade, 1.0), (grade_data, assigned_grade, weight)
    )
    print(res)
    params = res.x
    print("LogScore: ", log_score(grade_data, skewnorm, *params))
    print("mode delta: ", mode_delta(params[1], grade_data, assigned_grade))
    return params
    # except RuntimeError:
    #    print("optimsation failed")
    #    grade_peak = grades[grade_histogram.argmax()]
    #    floc = newton(f, assigned_grade, args=(grade_data, grade_peak))
    #    return skewnorm.fit(grade_data, floc=floc)


def plot_model(grades, params):
    xs = np.linspace(1, 40, 1000)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    pdf = skewnorm.pdf(xs, *params)
    ax1.plot(xs, pdf)
    ax2.bar(np.arange(1, 40), grades, alpha=0.2)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax1.set_title(
        f"shape: {params[0]:.2f}, loc: {params[1]:.2f}, scale: {params[2]:.2f}"
    )


def get_popular(session: KilterAPI, minimum_ascents: int, angle: int | None):
    climbs = session.tables["climbs"]
    climb_stats = session.tables["climb_stats"].set_index(["climb_uuid", "angle"])
    all_climbs = climbs.join(
        climb_stats,
        on=["uuid", "angle"],
        rsuffix="_r",
    )[["uuid", "angle", "name", "ascensionist_count", "setter_username"]]
    popular_climbs = all_climbs[all_climbs["ascensionist_count"] >= minimum_ascents]
    if angle is not None:
        popular_climbs = popular_climbs[popular_climbs["angle"] == angle]
    return popular_climbs


def fit(row, session):
    hist = grade_histogram(session, row.uuid, int(row.angle))
    params = fit_grade_curve(hist)
    return pd.Series(params, index=["shape", "loc", "scale"])


def get_benchmarks(
    session: KilterAPI, minimum_ascents: int, angle: int
) -> pd.DataFrame:
    popular = get_popular(session, minimum_ascents, angle)
    params_df = popular.apply(fit, args=(session,), axis=1)

    grades_df = session.tables["difficulty_grades"]
    params_df["mode"] = params_df.apply(lambda p: skewnorm_mode(*p), axis=1)
    joined = popular.join(params_df)
    joined = joined.join(
        grades_df["boulder_name"].str.split("/").str[1],
        on=joined["loc"].round().astype(int),
    ).rename(columns={"verm_name": "grade"})
    return joined
