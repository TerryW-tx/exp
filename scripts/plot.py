from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot summary CSV for vSFC experiments")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Path to summary.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    parser.add_argument("--x", type=str, default="metric", help="Column name for x-axis")
    parser.add_argument("--y", type=str, default="mean", help="Column name for y-axis")
    parser.add_argument("--yerr", type=str, default="std", help="Column name for error bar")
    parser.add_argument("--title", type=str, default="Experiment Summary", help="Figure title")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise ValueError("Summary CSV is empty. Please check metrics implementation and runs.")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # 占位图类型：误差棒柱状图。后续可根据配置扩展为折线/箱线图等。
    ax.bar(
        df[args.x].astype(str),
        df[args.y].astype(float),
        yerr=df[args.yerr].astype(float) if args.yerr in df.columns else None,
        capsize=4,
    )
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)
    ax.set_title(args.title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
