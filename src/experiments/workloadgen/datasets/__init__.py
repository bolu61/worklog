from pathlib import Path

from experiments.datasets.csv import masked_csv_dataset

from .synthetic import masked_process_dataset


def easy_synthetic_dataset(key):
    return masked_process_dataset(
        key=key,
        size=10_000,
        interleaving=4,
        states=4,
        alphabet=4,
        shape=1,
        length=10,
    )


def apache_james(key):
    return masked_csv_dataset(
        key=key,
        path=Path(__file__).parent / "apache_james_load_100k.csv",
        size=10_000,
        length=10,
    )


def openmrs(key):
    return masked_csv_dataset(
        key=key,
        path=Path(__file__).parent / "openmrs_load_100k.csv",
        size=10_000,
        length=10,
    )


def google_borg(key):
    return masked_csv_dataset(
        key=key,
        path=Path(__file__).parent / "google_borg_load_100k.csv",
        size=10_000,
        length=10,
    )
