import copy
from functools import cache
import multiprocessing

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skewnorm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.kilterbench.kilter_api import KilterAPI
from src.kilterbench.stats import histogram_to_data, skewnorm_mode, log_score, rescale_peak


@cache
def grade_histogram(session: KilterAPI, climb_id: str, angle: int) -> np.ndarray:
    stats = session.get_climb_stats(climb_id, angle)
    grades = np.zeros(39, dtype=int)
    for s in stats["difficulty"]:
        grades[s["difficulty"] - 1] = s["count"]
    return grades


def fit_grade_curve(grade_histogram: np.ndarray) -> tuple[float, float, float]:
    def mode_delta(x: float, data: np.ndarray, target_mode: float) -> float:
        params = skewnorm.fit(data, floc=x)
        return skewnorm_mode(*params) - target_mode

    def opt_func(params, data, target_mode, weight) -> float:
        params[2] = max(params[2], 1e-8)
        shape, loc, scale = params
        delta = mode_delta(loc, data, target_mode)
        # score = crps(data, skewnorm, *params)
        score = log_score(data, skewnorm, *params)
        return delta * delta + weight * score

    grades = np.arange(1, 40)
    idxmax = grade_histogram.argmax()
    assigned_grade: int = grades[idxmax]

    target_assigned_proportion = 0.5
    grade_histogram = rescale_peak(grade_histogram.copy(), target_assigned_proportion)

    grade_data = histogram_to_data(grades, grade_histogram)
    params = skewnorm.fit(grade_data)
    weight = 3
    res = minimize(
        opt_func, (0, assigned_grade, 1.0), (grade_data, assigned_grade, weight), "BFGS"
    )
    params = tuple(res.x)
    return params


def plot_model(
    grades: np.ndarray, params: tuple[float, float, float], label: str
) -> Figure:
    xs = np.linspace(1, 40, 1000)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    pdf = skewnorm.pdf(xs, *params)
    ax1.plot(xs, pdf)
    ax2.bar(np.arange(1, 40), grades, alpha=0.2)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.suptitle(label)
    ax1.set_title(
        "\n".join(
            [
                f"shape: {params[0]:.2f}, loc: {params[1]:.2f}, scale: {params[2]:.2f}",
                f"mode: {skewnorm_mode(*params):.2f}",
            ]
        ),
        size=10,
    )
    return fig


def get_popular(
    session: KilterAPI, minimum_ascents: int, angle: int | None = None
) -> pd.DataFrame:
    climbs = session.tables["climbs"].set_index("uuid_upper")
    climb_stats = session.tables["climb_stats"]
    all_climbs = climb_stats.join(
        climbs,
        on="climb_uuid_upper",
        rsuffix="_r",
    )[["climb_uuid", "angle", "name", "ascensionist_count", "setter_username"]]
    popular_climbs = all_climbs[all_climbs["ascensionist_count"] >= minimum_ascents]
    if angle is not None:
        popular_climbs = popular_climbs[popular_climbs["angle"] == angle]
    return popular_climbs


def fit(row, session) -> pd.Series:
    hist = grade_histogram(session, row.climb_uuid, int(row.angle))
    params = fit_grade_curve(hist)
    return pd.Series(params, index=["shape", "loc", "scale"])


def _parallel_hist(uuid: str, angle: int):
    return grade_histogram(session, uuid, int(angle))


def _worker_init(_session: KilterAPI):
    global session
    session = _session


def get_benchmarks(
    session: KilterAPI, minimum_ascents: int, angle: int | None = None
) -> pd.DataFrame:
    popular = get_popular(session, minimum_ascents, angle)

    session = copy.copy(session)
    session.reset()

    with multiprocessing.Pool(initializer=_worker_init, initargs=(session,)) as pool:
        histograms = pool.starmap(
            _parallel_hist,
            zip(
                popular["climb_uuid"].to_list(),
                popular["angle"].to_list(),
            ),
        )
        print("Histograms retrieved...")
        params = pool.map(fit_grade_curve, histograms)
        params_df = pd.DataFrame(
            params, index=popular.index, columns=["shape", "loc", "scale"]
        )
        print("Curves Fitted")

    grades_df = session.difficulty_grades
    params_df["mode"] = params_df.apply(lambda p: skewnorm_mode(*p), axis=1)
    joined = popular.join(params_df)
    joined = joined.join(
        grades_df["boulder_name"].str.split("/").str[1],
        on=joined["mode"].round().astype(int),
    ).rename(columns={"boulder_name": "grade"})
    return joined
