from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch import Generator
import torch.nn as nn
from dataset import ADFALDDataset, load_dataset

# Load ADFA-LD dataset into PyTorch
train_normal_path, valid_normal_path, attack_data_path = load_dataset()

normal_dataset = ConcatDataset([
    ADFALDDataset(train_normal_path, 0),
    ADFALDDataset(valid_normal_path, 0)
])
attack_dataset = ADFALDDataset(attack_data_path, 0)

# Split into train and test
TRAIN_RATIO: float = 0.8

train_normal_size = int(TRAIN_RATIO * len(normal_dataset))
valid_normal_size = len(normal_dataset) - train_normal_size
train_normal_subset, valid_normal_subset = random_split(
    normal_dataset, [train_normal_size, valid_normal_size], generator=Generator()#.manual_seed(42) # for reproducibility
)

train_attack_size = int(TRAIN_RATIO * len(attack_dataset))
valid_attack_size = len(attack_dataset) - train_attack_size
train_attack_subset, valid_attack_subset = random_split(
    attack_dataset, [train_attack_size, valid_attack_size], generator=Generator()#.manual_seed(42) # for reproducibility
)

# Create the final train and validation datasets
train_dataset = ConcatDataset([train_normal_subset, train_attack_subset])
valid_dataset = ConcatDataset([valid_normal_subset, valid_attack_subset])

# Create DataLoaders
BATCH_SIZE: int = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)