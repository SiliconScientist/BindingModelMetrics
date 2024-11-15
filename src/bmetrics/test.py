from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool





def run():
    # I want an if statement that checks if the output file already exists
    # if not os.path.exists(config.output_path):
    #     dataset = LmdbDataset({"src": config.lmdb_path})
    #     atoms_list = []
    #     dft_energy_list = []
    #     for lmdb_data_object in tqdm(dataset, desc=f"Extracting atoms and energies"):
    #         atoms = lmdb_to_atoms(lmdb_data_object=lmdb_data_object)
    #         energy = lmdb_data_object.y_relaxed
    #         atoms_list.append(atoms)
    #         dft_energy_list.append(energy)
    #     df = pl.DataFrame({"dft": dft_energy_list})
    #     calculators = load_calculators(config.model_names, config.model_keys)
    #     for key in calculators.keys():
    #         ml_energy_list = []
    #         for atoms in tqdm(atoms_list, desc=f"Calculating {key} energies"):
    #             atoms.calc = calculators[key]
    #             energy = atoms.get_potential_energy()
    #             ml_energy_list.append(energy)
    #         df = df.with_columns([pl.Series(ml_energy_list).alias(key)])
    #     df.write_parquet(config.output_path)
    # df = pl.read_parquet(config.output_path)
    # print(df)

    # ground_truth = np.array(df["dft_energy"].to_list())

    # scaled_y_lists = []
    # for column in df.columns:
    #     if column != "dft_energy":
    #         predicted_energies = np.array(df[column].to_list())
    #         slope, intercept = np.polyfit(x=predicted_energies, y=ground_truth, deg=1)
    #         regression_line = slope * predicted_energies + intercept
    #         scaled_y_lists.append(regression_line)
    # averages = np.average(np.array(scaled_y_lists), axis=0)
    # std_dev = np.std(np.array(scaled_y_lists), axis=0)
    # plt.errorbar(
    #     ground_truth,
    #     averages,
    #     yerr=std_dev,
    #     fmt="o",
    #     color="#1f77b4",
    #     ecolor="black",
    #     capsize=5,
    # )
    # plt.xlabel("Ensemble Energy")
    # plt.ylabel("Ground Truth Energy", rotation=0, labelpad=40)
    # plt.plot(
    #     [min(ground_truth), max(ground_truth)],
    #     [min(averages), max(averages)],
    #     color="black",
    #     linestyle="--",
    # )
    # plt.savefig("../../data/output/regression.png")
    # plt.show()


if __name__ == "__main__":
    run()