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

'''
[{
    "query": "...",
    "topk": [{"id": source_name, "text": "...", "score": 0.1}, 
                {'id': 'Kirsten Gillibrand', 'text': 'Kirsten Gillibrand is a Senator.', 'score': 0.57928383}
                ],
    "ground_truth": "..."
}, {...}, {...}]
'''
            


def main(args):
    dr = DenseRetriever.load(
        index_name=args.index,
        device=args.device,
        transformers_cache_dir=args.huggingface_cache_dir,
    )
    k = args.cutoff
    print(f"Using {args.device} as device")

    included_doc = []
    # test_dir = '../data_preprocessed/test'
    # for filename in tqdm(os.listdir(test_dir)):
    #     file_path = os.path.join(test_dir, filename)
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         contents = json.load(f)
    #         for content in contents:
    #             included_doc.extend(content['sources'].keys())

    

    test_path = './test_set.json'
    included_doc_path = './test_id.json'
    with open(test_path, 'r') as f:
        contents = json.load(f)

    with open(included_doc_path, 'r') as f:
        included_doc = json.load(f)
    
    # for content in contents:
    #     included_doc.extend(content['obscured_sources'].keys())

    print(f'numher of sources: {len(included_doc)}')

    search_results = []
    # for filename in tqdm(os.listdir(test_dir)):
    #     file_path = os.path.join(test_dir, filename)
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         contents = json.load(f)

    for content in tqdm(contents):
        sources = content['obscured_sources']
        questions = content['questions']
        for question, source in zip(questions.items(), sources.items()):
            q = question[1]
            source_name = source[0]
            source_obscured = source[1]
            topk = dr.search(q, include_id_list=included_doc, cutoff=k, return_docs=True)
            # topk = dr.search(q, cutoff=k, return_docs=True)
            

            for result in topk:
                result["score"] = str(result["score"])

            search_results.append({
                "query": q,
                "topk": topk,
                "ground_truth_source": source_name,
                "ground_truth_obscured_sources": source_obscured
            })

    print(search_results[0])
    

    # # msearch
    # all_questions = []
    # all_sources = []

    # for filename in tqdm(os.listdir(directory)):
    #     file_path = os.path.join(directory, filename)
        
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         contents = json.load(f)
    #     for content in contents:
    #         questions = content['questions']
    #         sources = content['obscured_sources']
    #         for question, source in zip(questions.items(), sources.items()):
    #             all_questions.append(question[1])
    #             all_sources.append(source[1])
    # queries = []

    # for question, source in zip(all_questions, all_sources):
    #     queries.append({
    #         "id": source,
    #         "text": question
    #     })
    # search_results = dr.msearch(
    #                     queries=queries,
    #                     cutoff=10,
    #                     batch_size=32
    #                 )
    

    print("writing outputs")
    with open(f'../data_baseline/vanilla_{k}.json', 'w') as f:
        json.dump(search_results, f, indent=2)


    print("DONE!!!")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Salesforce/SFR-Embedding-2_R')
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--index', type=str, default='obscured_sources')
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return for each query')
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default='/project/jonmay_231/spangher/huggingface_cache',
        help="Path to the directory containing HuggingFace cache"
    )
    parser.add_argument('--cutoff', type=int, default=10)
    args = parser.parse_args()
    main(args)
