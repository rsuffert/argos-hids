from torch.utils.data import DataLoader
from dataset import ADFALDDataset, load_dataset

train_normal_path, valid_normal_path, attack_data_path = load_dataset()

train_normal = ADFALDDataset(train_normal_path, 0)
valid_normal = ADFALDDataset(valid_normal_path, 0)
attack_data  = ADFALDDataset(attack_data_path, 1)

print("Normal instances for training:", len(train_normal))
print("Normal instances for validation:", len(valid_normal))
print("Attack instances:", len(attack_data))