from __future__ import annotations

import sys

import click
import pandas as pd

from datasetsanity.core import DatasetSanity


@click.group()
def main() -> None:
    """DatasetSanity — sanity checks for ML datasets."""


@main.command()
@click.argument("csv_file")
@click.option("--target", required=True, help="Name of the target column.")
@click.option(
    "--task",
    default="classification",
    show_default=True,
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    help="ML task type.",
)
@click.option("--output", default=None, help="Optional path to write the JSON report.")
def check(csv_file: str, target: str, task: str, output: str) -> None:
    """Run sanity checks on CSV_FILE."""
    df = pd.read_csv(csv_file)
    checker = DatasetSanity(df, target=target, task=task)
    report = checker.run()
    report.summary()

    if output:
        report.to_json(output)
        click.echo(f"Report written to {output}")

    all_passed = (
        report.missing_values.passed
        and report.class_imbalance.passed
        and report.leakage.passed
    )
    if not all_passed:
        sys.exit(1)
