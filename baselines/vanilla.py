from dense_retriever import MyDenseRetriever
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
            index_name="sources",
            model=args.embedding_model,
            normalize=True,
            max_length=args.max_seq_length,
            embedding_dim=args.embedding_dim,
            device=args.device,
            use_ann=True,
        )


    source_files = ["sources_data_70b__0_10000.json", 
                        "sources_data_70b__100000_110000.json", 
                        "sources_data_70b__10000_20000.json", 
                        "sources_data_70b__110000_120000.json", 
                        "sources_data_70b__120000_130000.json",	
                        "sources_data_70b__200000_200100.json",		
                        "sources_data_70b__200000_205000.json",	
                        "sources_data_70b__205000_210000.json",	
                        "sources_data_70b__210000_220000.json",
                        "sources_data_70b__220000_230000.json",
                        "sources_data_70b__230000_240000.json",
                        "sources_data_70b__240000_250000.json",		
                        "sources_data_70b__310000_320000.json",
                        "sources_data_70b__320000_330000.json",
                        "sources_data_70b__330000_340000.json",
                        "sources_data_70b__80000_90000.json",	
                        "sources_data_70b__90000_100000.json"]
    collection = []

    for filename in source_files:
        full_path = "../data/" + filename + '.josn'
        with open(full_path, 'r') as f:
            data = json.load(f)
            for article in data:
                for id, summary in article['sources'].items():
                    new_source_embedding = {"id": id, "text": summary}
                    collection.append(new_source_embedding)
    dr.index(
        collection=collection, 
        show_progress=True, 
    )

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