import argparse
import os
import json
import torch
from dense_retriever import MyDenseRetriever
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
    dr = MyDenseRetriever.load(
        index_name=args.index,
        device=args.device
    )
    print(f"Using {args.device} as device")

    included_doc = []
    test_dir = '../data_preprocessed/test'
    for filename in tqdm(os.listdir(test_dir)):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                included_doc.extend(content['sources'].keys())

    print(f'test size: {len(included_doc)}')

    directory = '../data_preprocessed'
    search_results = []
    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = json.load(f)

        for content in contents:
            sources = content['obscured_sources']
            questions = content['questions']
            for question, source in zip(questions.items(), sources.items()):
                q = question[1]
                s = source[1]
                topk = dr.search(q, include_id_list=included_doc, cutoff=10, return_docs=True)
                search_results.append({
                    "query": q,
                    "topk": topk,
                    "ground_truth": s
                })
        print(f"finished processing {filename}")
    

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
    with open('../data_baselines/vanilla.json', 'w') as f:
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
    
    args = parser.parse_args()
    main(args)
