import argparse
import os
import json
from dense_retriever import MyDenseRetriever
from dr_search import search

# Set environment variables
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME
cwd = os.path.dirname(os.path.abspath(__file__))
os.environ['RETRIV_BASE_PATH'] = cwd

def main(args):
    # Load the existing index
    dr = MyDenseRetriever.load(
        index_name=args.index,
        model=args.embedding_model,
        normalize=True,
        max_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        device=args.device,
        use_ann=True,
    )
    '''
    # Load queries from a file
    with open(args.query_file, 'r') as f:
        queries = json.load(f)

    # Perform search for each query
    results = []
    for query in queries:
        search_results = dr.search(query, cutoff=args.top_k)
        results.append({
            "query": query,
            "results": search_results
        })

    # Save results to a file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Search completed. Results saved to {args.output_file}")

    '''
    print(dr.search('How does the halfRF system created by the BBC work?'))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model', type=str, default='Salesforce/SFR-Embedding-2_R')
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--index', type=int, default='sources')
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--query_file', type=str, required=True, help='Path to the file containing queries')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the search results')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return for each query')
    
    args = parser.parse_args()
    main(args)