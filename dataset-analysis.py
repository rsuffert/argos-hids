import kagglehub
import os

path = kagglehub.dataset_download("yianqaq/adfa-ld")
path = os.path.join(path, "ADFA-LD")

train_normal = os.path.join(path, "Training_Data_Master")
valid_normal = os.path.join(path, "Validation_Data_Master")
attack_data  = os.path.join(path, "Attack_Data_Master")

train_normal_count = sum(1 for entry in os.scandir(train_normal))
valid_normal_count = sum(1 for entry in os.scandir(valid_normal))
print(f"Normal sequences count: {train_normal_count + valid_normal_count}")

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
print(f"Attack types count per class (total = {total_attacks_count}):")
for attack_type, count in attack_types_count.items():
    print(f"\t{attack_type}: {count}")