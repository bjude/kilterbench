import pandas as pd

import argparse

from kilterbench import kilter_api
from kilterbench import benchmarks


def add_fit_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "fit", help="Fit statistical parameters to the climb grade distributions"
    )

    parser.add_argument(
        "--min_repeats",
        help="Minimum number of repeats to consider when identifying benchmarks",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save",
        help="Store the fitted distributions in a json file",
        action="store_true",
    )


def add_circuit_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("circuit", help="Create Circuits")
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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_fit_subparser(subparsers)
    add_circuit_subparser(subparsers)

    # Username and pass needed for fitting and creating circuits
    parser.add_argument("-u", "--username", help="Username", required=True)
    parser.add_argument("-p", "--password", help="Password", required=True)

    args = parser.parse_args()

    session = kilter_api.KilterAPI(args.username, args.password)
    if args.command == "bench":
        benches = benchmarks.get_benchmarks(session, 1000)
        if args.save:
            benches.to_json("benches.json")
    elif args.command == "circuit":
        print("Reading json")
        benches = pd.read_json("benches.json").sort_values("mode")
        for angle in benches["angle"].sort_values().unique():
            bench_mask = benches["shape"].abs() < args.max_skew
            angle_mask = benches["angle"] == angle
            uuids = benches[bench_mask & angle_mask]["climb_uuid"].to_list()
            circuit_name = f"{args.prefix} - {angle:>02}"
            print(f"Making circuit: '{circuit_name}' with {len(uuids)} climbs")
            circuit_id = session.make_new_circuit(circuit_name)
            session.set_circuit(circuit_id, uuids)


if __name__ == "__main__":
    main()
