import pandas as pd
import seaborn as sn

import argparse

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
        default=1000,
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
    parser.add_argument(
        "--max_skew",
        type=float,
        help="Maximum value of the shape parameter of the fitted skewed normal distributions",
        default=1.0,
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
        benches = benchmarks.get_benchmarks(session, args.min_repeats)
        benches.to_json("benches.json")
    if args.command == "circuit":
        print("Reading json")
        benches = pd.read_json("benches.json").sort_values("mode")
        session = kilter_api.KilterAPI(args.username, args.password)
        for angle in benches["angle"].sort_values().unique():
            bench_mask = benches["shape"].abs() < args.max_skew
            angle_mask = benches["angle"] == angle
            uuids = benches[bench_mask & angle_mask]["climb_uuid"].to_list()
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
