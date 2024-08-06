import argparse
import os
import json
import torch
from retriv import DenseRetriever, Encoder, SparseRetriever
from tqdm.auto import tqdm
from dr_search import search

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME
cwd = os.path.dirname(os.path.abspath(__file__))
os.environ['RETRIV_BASE_PATH'] = cwd


def main(args):
    dr = DenseRetriever.load(
        index_name=args.index,
        device=args.device,
        transformers_cache_dir=args.huggingface_cache_dir,
    )
    print(f"Using {args.device} as device")

    included_doc = []

    test_path = './test_set.json'
    with open(test_path, 'r') as f:
        contents = json.load(f)
    
    for content in contents:
        included_doc.extend(content['sources'].keys())

    print(f'test size: {len(included_doc)}')

    search_results = []

    for content in tqdm(contents):
        sources = content['obscured_sources']
        questions = content['questions']
        for question, source in zip(questions.items(), sources.items()):
            q = question[1]
            s = source[1]
            topk = dr.search(q, include_id_list=included_doc, cutoff=10, return_docs=True)

            for result in topk:
                result["score"] = str(result["score"])

            search_results.append({
                "query": q,
                "topk": topk,
                "ground_truth": s
            })

    print(search_results[0])
    
    print("writing outputs")
    with open('../data_baseline/source_aug.json', 'w') as f:
        json.dump(search_results, f, indent=2)

    print("DONE!!!")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Salesforce/SFR-Embedding-2_R')
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--index', type=str, default='augmented_sources')
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return for each query')
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default='/project/jonmay_231/spangher/huggingface_cache',
        help="Path to the directory containing HuggingFace cache"
    )
    args = parser.parse_args()
    main(args)
