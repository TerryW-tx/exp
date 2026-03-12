from __future__ import annotations

import argparse
from pathlib import Path

from vsfc_lab.mock_components import (
    build_mock_requests,
    build_mock_topology,
    save_scenario_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible vSFC scenario JSON files (topology + requests)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store generated scenario JSON files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds used to generate one scenario per seed.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10,
        help="Upper bound of sampled nodes per scenario (<=10).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("site-optus-melbCBD.csv"),
        help="Path to site CSV used for topology generation.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="scenario",
        help="Output file prefix. File pattern: <prefix>_seed<seed>.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite scenario file if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.max_nodes < 2:
        raise ValueError("--max-nodes must be >= 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    skipped_files: list[Path] = []

    for seed in args.seeds:
        topology = build_mock_topology(
            seed=seed,
            max_nodes=args.max_nodes,
            csv_path=args.csv_path,
        )
        requests = build_mock_requests(node_ids=topology.node_ids)

        out_file = args.output_dir / f"{args.prefix}_seed{seed}.json"
        if out_file.exists() and not args.overwrite:
            skipped_files.append(out_file)
            continue

        save_scenario_config(out_file, topology, requests)
        generated_files.append(out_file)

    print(f"Generated {len(generated_files)} scenario file(s).")
    for file in generated_files:
        print(f"  + {file}")

    if skipped_files:
        print(f"Skipped {len(skipped_files)} existing file(s). Use --overwrite to replace.")
        for file in skipped_files:
            print(f"  - {file}")


if __name__ == "__main__":
    main()
