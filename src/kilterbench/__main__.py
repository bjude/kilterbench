import pandas as pd
import seaborn as sn

import argparse
import os
import re
import warnings

from kilterbench import kilter_api
from kilterbench import benchmarks


def add_fit_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "fit", help="Fit statistical parameters to the climb grade distributions"
    )

    parser.add_argument("-u", "--username", help="Username", required=True)
    parser.add_argument("-p", "--password", help="Password", required=True)
    parser.add_argument(
        "--min_repeats",
        help="Minimum number of repeats to consider when identifying benchmarks",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--parallel",
        help="The number of cores to use when fitting ascent distributions. Pass 0 to use all available cores (default)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--angles",
        type=int,
        nargs="*",
        help="Angles to consider, by default all available angles will be fitted",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots of fitted grade curve for every climb",
    )


def add_circuit_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("circuit", help="Create Circuits")

    parser.add_argument("-u", "--username", help="Username", required=True)
    parser.add_argument("-p", "--password", help="Password", required=True)
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for generated circuits. Circuits will be names as '{prefix} - {angle}'",
        default="BM",
    )
    parser.add_argument(
        "--angles",
        type=int,
        nargs="*",
        help="Angles to consider, by default a circuit will be generated for all available angles",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max_skew",
        type=float,
        help="Maximum value of the shape parameter of the fitted skewed normal distributions. Incompatible "
        "with also specifying `--skew_range`",
        default=1.0,
    )
    group.add_argument(
        "--skew_range",
        type=float,
        nargs=2,
        metavar=("MIN_SKEW", "MAX_SKEW"),
        help="Range of the shape parameter of the fitted skewed normal distributions. Incompatible with also "
        "specifying `--max_skew`",
        default=None,
    )


def add_plot_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("plot", help="Plot climb statistics")
    parser.add_argument(
        "--summary",
        help="Plot the summary distributions for each angle",
        action="store_true",
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_fit_subparser(subparsers)
    add_circuit_subparser(subparsers)
    add_plot_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "fit":
        session = kilter_api.KilterAPI(args.username, args.password)
        cores = args.parallel if args.parallel > 0 else None
        benches = benchmarks.get_benchmarks(
            session, args.min_repeats, num_processes=cores, angles=args.angles or None
        )
        benches.to_json("benches.json")

        if args.save_plots:
            print("Saving Plots...")
            os.makedirs("benchmark_plots", exist_ok=True)
            for row in benches.itertuples():
                hist = benchmarks.grade_histogram(session, row.climb_uuid, row.angle)
                # NOTE: The scaled histogram is used here because if we dont clip the assigned grade then it
                # will completely dwarf the rest of the histogram. The value of 0.5 is hardcoded here to match
                # the value used in benchmarks.fit_grade_curve
                hist = benchmarks.rescale_peak(hist, 0.5)
                label = f"{row.name} @ {row.angle}° - {row.grade}"
                params = (row.shape, row.loc, row.scale)
                fig = benchmarks.plot_model(hist, params, label)
                # Remove any invalid characters from the filename
                clean_name  = re.sub(r'[\\/*?:"<>|]', "", row.name).strip()
                fig.savefig(f"benchmark_plots/{clean_name}_{row.angle}.png")

    if args.command == "circuit":
        print("Reading json")
        benches = pd.read_json("benches.json").sort_values("mode")
        session = kilter_api.KilterAPI(args.username, args.password)
        for angle in benches["angle"].sort_values().unique():
            if args.angles and angle not in args.angles:
                # Angles passed on the command line and does not include the angle being processed, skip
                continue
            skew_range = args.skew_range or (-args.max_skew, args.max_skew)
            bench_mask = benches["shape"] > skew_range[0]
            bench_mask &= benches["shape"] < skew_range[1]
            angle_mask = benches["angle"] == angle
            uuids = benches[bench_mask & angle_mask]["climb_uuid"].to_list()
            if len(uuids) == 0:
                warnings.warn(
                    f"Angle {angle} contains no climbs. This may be because `kilterbench fit` was called without "
                    "requesting this angle, or other filtering parameters like the maximum skew or minimum number "
                    "of repeats may screened out all climbs."
                )
                continue
            circuit_name = f"{args.prefix} - {angle:>02}"
            print(f"Making circuit: '{circuit_name}' with {len(uuids)} climbs")
            circuit_id = session.make_new_circuit(circuit_name)
            session.set_circuit(circuit_id, uuids)
    elif args.command == "plot":
        benches = pd.read_json("benches.json").sort_values("mode")
        scale_lim = (0, 3)
        shape_lim = (-3, 3)

        benches["shape_clip"] = benches["shape"].clip(*shape_lim)
        benches["scale_clip"] = benches["scale"].clip(*scale_lim)

        os.makedirs("plots", exist_ok=True)
        for angle in sorted(benches["angle"].unique()):
            angle_mask = benches["angle"] == angle
            angle_benches = benches[angle_mask]
            sn.jointplot(
                angle_benches,
                x="shape",
                y="scale",
                kind="scatter",
                xlim=shape_lim,
                ylim=scale_lim,
                marginal_kws={"binrange": shape_lim, "bins": 20},
            ).savefig(f"plots/bench_summary_{angle:>02}.png")


if __name__ == "__main__":
    main()
