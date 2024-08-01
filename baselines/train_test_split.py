import os
import shutil
import random

def split_data(source_folder, train_folder, test_folder, test_ratio=0.2):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    print(f"Total number of files: {len(files)}")
    random.shuffle(files)
    
    test_size = int(len(files) * test_ratio)
    
    test_files = files[:test_size]
    train_files = files[test_size:]
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    for f in test_files:
        shutil.copy(os.path.join(source_folder, f), os.path.join(test_folder, f))
        
    for f in train_files:
        shutil.copy(os.path.join(source_folder, f), os.path.join(train_folder, f))

    print(f"Copied {len(test_files)} files to {test_folder}")
    print(f"Copied {len(train_files)} files to {train_folder}")


if __name__ == "__main__":
    source_folder = '../data_preprocessed'
    train_folder = '../data_preprocessed/train'
    test_folder = '../data_preprocessed/test'

    split_data(source_folder, train_folder, test_folder)
