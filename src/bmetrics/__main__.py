import toml
import time
import pickle

# from bmetrics.config import Config
from ase.db import connect
from ase.visualize import view
from fairchem.core.common.relaxation.ase_utils import OCPCalculator


def main():
    # config = Config(**toml.load("config.toml"))

    # Load the data from the pickle file
    with open(
        "/Users/averyhill/Github/BindingModelMetrics/data/h_saas/atoms_binding_data.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)
    checkpoint_path = "/Users/averyhill/Github/BindingModelMetrics/models/schnet_all.pt"
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
    # data is a list of tuples (atoms, binding_energy)
    atoms_list, binding_energies = zip(*data)
    atoms = atoms_list[0]
    energy = binding_energies[0]
    atoms.calc = calc
    ml_energy = atoms.get_potential_energy()
    print("energy: ", energy)
    print("ml_energy: ", ml_energy)
    print("done")


if __name__ == "__main__":
    main()
