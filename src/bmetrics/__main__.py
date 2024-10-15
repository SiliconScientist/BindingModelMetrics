import toml
import time
import pickle
import os
import polars as pl
import numpy as np

from bmetrics.config import Config
from ase.db import connect
from ase.geometry import wrap_positions
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.visualize import view
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
import matplotlib.pyplot as plt
import csv
from fairchem.core.datasets import LmdbDataset


def load_calculators(model_names: list[str], model_keys) -> dict[str, OCPCalculator]:
    calculators = {}
    names_tags = zip(model_names, model_keys)
    for name, tag in names_tags:
        checkpoint_path = model_name_to_local_file(
            model_name=name,
            local_cache="/tmp/fairchem_checkpoints/",
        )
        calc = OCPCalculator(checkpoint_path=checkpoint_path, seed=123)
        calculators[tag] = calc
    return calculators


def lmdb_to_atoms(lmdb_data_object) -> Atoms:
    atomic_symbols = [
        chemical_symbols[int(atomic_number)]
        for atomic_number in lmdb_data_object.atomic_numbers
    ]
    positions = lmdb_data_object.pos_relaxed
    cell = lmdb_data_object.cell[0]
    atoms = Atoms(
        atomic_symbols, positions=positions, cell=cell, pbc=[True, True, True]
    )
    return atoms


def main():
    config = Config(**toml.load("config.toml"))

    # I want an if statement that checks if the output file already exists
    if not os.path.exists(config.output_path):
        dataset = LmdbDataset({"src": config.lmdb_path})
        atoms_list = []
        dft_energy_list = []
        for lmdb_data_object in dataset:
            atoms = lmdb_to_atoms(lmdb_data_object=lmdb_data_object)
            energy = lmdb_data_object.y_relaxed
            atoms_list.append(atoms)
            dft_energy_list.append(energy)
        df = pl.DataFrame({"dft": dft_energy_list})
        calculators = load_calculators(config.model_names, config.model_keys)
        for key in calculators.keys():
            ml_energy_list = []
            for atoms in atoms_list:
                atoms.calc = calculators[key]
                energy = atoms.get_potential_energy()
                ml_energy_list.append(energy)
            df = df.with_columns([pl.Series(ml_energy_list).alias(key)])
        df.write_parquet(config.output_path)

    df = pl.read_parquet(config.output_path)
    print(df)

    ground_truth = np.array(df["dft_energy"].to_list())

    scaled_y_lists = []
    for column in df.columns:
        if column != "dft_energy":
            predicted_energies = np.array(df[column].to_list())
            slope, intercept = np.polyfit(x=predicted_energies, y=ground_truth, deg=1)
            regression_line = slope * predicted_energies + intercept
            scaled_y_lists.append(regression_line)
    averages = np.average(np.array(scaled_y_lists), axis=0)
    std_dev = np.std(np.array(scaled_y_lists), axis=0)
    plt.errorbar(
        ground_truth,
        averages,
        yerr=std_dev,
        fmt="o",
        color="#1f77b4",
        ecolor="black",
        capsize=5,
    )
    plt.xlabel("Ensemble Energy")
    plt.ylabel("Ground Truth Energy", rotation=0, labelpad=40)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(averages), max(averages)],
        color="black",
        linestyle="--",
    )
    plt.savefig("../../data/output/regression.png")
    plt.show()


if __name__ == "__main__":
    main()
