import argparse

from kilterbench import kilter_api
from kilterbench import benchmarks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-u", "--username", help="Username", required=True)
    parser.add_argument("-p", "--password", help="Password", required=True)

    parser.add_argument(
        "--min_repeats",
        help="Minimum number of repeats to consider when identifying benchmarks",
        default=1000,
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for generated circuits. Circuits will be names as '{prefix} - {angle}'",
        default="BM",
    )
    parser.add_argument(
        "--angles",
        nargs="*",
        help="Angles to consider, by default a circuit will be generated for all available angles",
    )

    parser.add_argument(
        "max_skew",
        help="Maximum value of the shape parameter of the fitted skewed normal distributions",
        default=1.0,
    )

    args = parser.parse_args()

    session = kilter_api.KilterAPI(args.username, args.password)
    benches = benchmarks.get_benchmarks(session, 1000)

    for angle in benches["angle"].sort_values().unique():
        bench_mask = benches["shape"].abs() < args.max_skew
        angle_mask = benches["angle"] == angle
        uuids = benches[bench_mask & angle_mask]["climb_uuid"].to_list()
        circuit_id = session.make_new_circuit(f"{args.prefix} - {angle}")
        session.set_circuit(circuit_id, uuids)


if __name__ == "__main__":
    main()
