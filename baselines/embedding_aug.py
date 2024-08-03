from dense_retriever import MyDenseRetriever
from tqdm.auto import tqdm
import argparse
import json
import torch
import os


data_path = "../data"
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME
cwd = os.path.dirname(os.path.abspath(__file__))
os.environ['RETRIV_BASE_PATH'] = cwd

def main(args):

    dr = MyDenseRetriever(
            index_name="augmented_sources",
            model=args.embedding_model,
            normalize=True,
            max_length=args.max_seq_length,
            embedding_dim=args.embedding_dim,
            device=args.device,
            use_ann=True,
        )

    directory = '../source_augmented'
    augmented_source_files = [file_name for file_name in os.listdir(directory)]
    collection = []

    for filename in tqdm(augmented_source_files):
        full_path = os.path.join(directory, filename)
        with open(full_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'{filename} is empty or not well formatted')
                continue
            for article in data:
                ids = article['obscured_sources'].keys()
                for id in ids:
                    new_source_embedding = {"id": id, "text": article['obscured_sources'][id] + "  " + article['augmented_sources'][id]}
                    collection.append(new_source_embedding)

                # for id, augmentation in zip(article['augmented_sources'].items(), article['obscured_sources'].items()):
                #     new_source_embedding = {"id": id, "text": augmentation}
                #     collection.append(new_source_embedding)

    print("number of documents:", len(collection))
    print(collection[0])
    dr.index(
        collection=collection, 
        show_progress=True, 
    )

    print("DONE!!!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_model', 
        type=str, 
        default='Salesforce/SFR-Embedding-2_R',
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=None,  # 4096
        help="The dimension of the embeddings"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,  # 32768,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for inference"
    )
    args = parser.parse_args()
    main(args)