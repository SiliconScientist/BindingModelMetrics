from fairchem.core.preprocessing import AtomsToGraphs
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from sklearn.model_selection import train_test_split
import lmdb
import pickle
from tqdm import tqdm
import torch
import os


def filter_reactions(reactions: list[dict], bound_sites: list[str]) -> list[Atoms]:
    """
    Filter out any reaction not specified in the products lists
    """
    systems = []
    for reaction in tqdm(reactions, desc="filter_reactions"):
        for key in reaction["reactionSystems"]:
            if key in bound_sites:
                atoms = reaction["reactionSystems"][key]
                atoms.calc = SinglePointCalculator(
                    atoms, energy=reaction["reactionEnergy"]
                )
                systems.append(atoms)
    return systems


a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,  # False for test data
    r_forces=False,  # False for test data
    r_distances=False,
    r_fixed=True,
)

with open("data/mamun/reactions.pkl", "rb") as f:
    reactions = pickle.load(f)

bound_sites = [
    "Ostar",
    "Cstar",
    "Hstar",
    "CH3star",
    "Nstar",
    "CH2star",
    "CHstar",
    "NHstar",
    "OHstar",
    "H2Ostar",
    "SHstar",
]
raw_data = filter_reactions(reactions=reactions, bound_sites=bound_sites)

tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)
train, temp = train_test_split(data_objects, test_size=0.1, random_state=123)
test, temp = train_test_split(temp, test_size=0.5, random_state=123)
val, cal = train_test_split(temp, test_size=0.5, random_state=123)
train_val_cal_test = {
    "train": train,
    "validation": val,
    "calibration": cal,
    "test": test,
}

for split, data_objects in train_val_cal_test.items():
    output_dir = "data/mamun/" + split
    os.makedirs(output_dir, exist_ok=True)
    file_path = output_dir + "/data.lmdb"
    db = lmdb.open(
        file_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
        # assign sid
        data.sid = torch.LongTensor([0])
        # assign fid
        data.fid = torch.LongTensor([fid])
        # assign tags, if available
        data.tags = torch.LongTensor(tags)
        # Filter data if necessary
        # FAIRChem filters adsorption energies > |10| eV and forces > |50| eV/A
        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            # print("no neighbors", traj_path)
            continue
        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
    txn.commit()
    db.sync()
    db.close()
