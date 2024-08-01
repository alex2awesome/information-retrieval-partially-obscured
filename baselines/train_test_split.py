import os
import shutil
import random
from tqdm.auto import tqdm

def split_data(source_folder, train_folder, test_folder, test_ratio=0.2):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    print(f"Total number of files: {len(files)}")
    random.shuffle(files)

    test_size = int(len(files) * test_ratio)
    test_files = files[:test_size]
    train_files = files[test_size:]
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    print("Copying files to test folder...")
    for file in tqdm(test_files, desc="Copying test files", unit="file"):
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))
        
    print("Copying files to train folder...")
    for file in tqdm(train_files, desc="Copying train files", unit="file"):
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    print(f"Copied {len(test_files)} files to {test_folder}")
    print(f"Copied {len(train_files)} files to {train_folder}")


if __name__ == "__main__":
    source_folder = '../data_preprocessed'
    train_folder = '../data_preprocessed/train'
    test_folder = '../data_preprocessed/test'

    split_data(source_folder, train_folder, test_folder)
