import kagglehub
import os
from typing import Tuple
from torch.utils.data import Dataset
import torch

class ADFALDDataset(Dataset):
    """
    Represents a subset of the ADFA-LD dataset entries with a given label.

    Args:
    root_dir (str): The root directory where the entries are located.
    label (int): The label for the entry, where 0 represents non-malicious and 1 represents malicious.
    """
    def __init__(self, root_dir: str, label: int):
        self.samples = []
        self.label   = label
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(dirpath, fname)
                with open(path) as f:
                    calls = list(map(int, f.read().split()))
                    self.samples.append((
                        torch.tensor(calls, dtype=torch.long),
                        label
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def load_dataset() -> Tuple[str, str, str]:
    """
    Loads the ADFA-LD dataset.

    Returns:
        Tuple[str, str, str]: Paths to the training, validation, and attack data directories, respectively.
    """
    path = kagglehub.dataset_download("yianqaq/adfa-ld")
    path = os.path.join(path, "ADFA-LD")

    train_normal = os.path.join(path, "Training_Data_Master")
    valid_normal = os.path.join(path, "Validation_Data_Master")
    attack_data  = os.path.join(path, "Attack_Data_Master")

    return train_normal, valid_normal, attack_data

def print_stats():
    """
    Prints statistics about the ADFA-LD dataset, including counts of normal sequences 
    and attack types.
    """
    train_normal, valid_normal, attack_data = load_dataset()

    print("ADFA-LD dataset statistics:")
    train_normal_count = sum(1 for entry in os.scandir(train_normal))
    valid_normal_count = sum(1 for entry in os.scandir(valid_normal))
    print(f"\tNormal sequences count: {train_normal_count + valid_normal_count}")

    attack_types_count = {
        "Adduser": 0,
        "Hydra_FTP": 0,
        "Hydra_SSH": 0,
        "Java_Meterpreter": 0,
        "Meterpreter": 0,
        "Web_Shell": 0
    }

    attack_types = attack_types_count.keys()

    for entry in os.scandir(attack_data):
        if not entry.is_dir():
            continue

        dir_name = entry.name
        attack_type = None
        for at in attack_types:
            if dir_name.startswith(at):
                attack_type = at
                break

        if not attack_type:
            continue

        attack_types_count[attack_type] += 1

    total_attacks_count = sum(attack_types_count.values())
    print(f"\tAttack types count per class (total = {total_attacks_count}):")
    for attack_type, count in attack_types_count.items():
        print(f"\t\t{attack_type}: {count}")

if __name__ == "__main__":
    print_stats()