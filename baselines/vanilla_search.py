import argparse
import os
import json
import torch
from dense_retriever import MyDenseRetriever
from dr_search import search

# Set environment variables
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME
cwd = os.path.dirname(os.path.abspath(__file__))
os.environ['RETRIV_BASE_PATH'] = cwd

def query_search(dr, contents):
    search_results = []
    for content in contents:
        sources = content['sources']
        questions = content['questions']
        for question, source in zip(questions.items(), sources.items()):
            q = question[1]
            s = source[1]
            print("question", type(q), q)
            print("source", s)
            topk = dr.search(question, cutoff=args.top_k)
            search_results.append({
                "query": q,
                "topk": topk,
                "ground_truth": s
            })


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
    # Load the existing index
    dr = MyDenseRetriever.load(
        index_name=args.index
    )
    print(dr.encoder)
    print(dr.encoder.embedding_dim)

    directory = '../data_preprocessed'
    search_results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            contents = json.load(f)
        search_result = query_search(dr, contents)
        search_results.extend(search_result)

    with open('../data_baselines/vanilla.json', 'w') as f:
        json.dump(search_results, f, indent=2)

    # print(dr.search("What is BBC?", cutoff=10))

    print("DONE!!!")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Salesforce/SFR-Embedding-2_R')
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--index', type=str, default='sources')
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return for each query')
    
    args = parser.parse_args()
    main(args)
