import os
import shutil
import random
import json
from tqdm.auto import tqdm

def split_data(source_folder, train_folder, test_folder, test_ratio=0.2):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    print(f"Total number of files: {len(files)}")
    random.shuffle(files)

    test_size = int(len(files) * test_ratio)
    test_files = files[:test_size]
    train_files = files[test_size:]
    
    # Check if train and test folders already exist and are not empty
    if os.path.exists(train_folder) and os.path.exists(test_folder) and \
       len(os.listdir(train_folder)) > 0 and len(os.listdir(test_folder)) > 0:
        print("Train and test folders already exist and contain files. Skipping split.")
        return
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

def generate_id(folder, setname):
    included_doc = []
    for filename in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                contents = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'{filename} is empty or not well formatted')
                continue
            for content in contents:
                included_doc.extend(content['sources'].keys())

    json_string = json.dumps(included_doc)
    outputname = ''
    if setname == 'test':
        outputname = 'test_id.json'
    else:
        outputname = 'train_id.json'

    with open(outputname, 'w') as f:
        f.write(json_string)

def generate_set(folder, setname):
    jsonfile = []
    ids = []
    for filename in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                contents = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'{filename} is empty or not well formatted')
                continue
            for content in contents:
                sources = content['obscured_sources']
                questions = content['questions']
                for source_name in questions:
                    if ":" in questions[source_name]:
                        questions[source_name] = questions[source_name].split(":")[1]
                    questions[source_name].replace("Here's a possible question that could have elicited this information:", "")
                    questions[source_name].replace("Here's a possible question that could have elicited this response:", "")
                    questions[source_name].replace("\n", "")
                    ids.append(source_name + '_' + content['article_url'])
                jsonfile.append({
                    'obscured_sources': sources,
                    'questions': questions
                })
    json_string = json.dumps(jsonfile)
    outputname = f'{setname}_set.json'
    with open(outputname, 'w') as f:
        f.write(json_string)
    with open(f'{setname}_id.json', 'w') as f:
        f.write(ids)


if __name__ == "__main__":
    source_folder = '../data_preprocessed_new'
    train_folder = '../data_preprocessed_new/train'
    test_folder = '../data_preprocessed_new/test'

    split_data(source_folder, train_folder, test_folder)
    # generate_id(test_folder, 'test')
    # generate_id(train_folder, 'train')
    generate_set(test_folder, 'test')
    generate_set(train_folder, 'train')

