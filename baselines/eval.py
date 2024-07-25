import argparse
import json

def precision(tp, fp):
    return tp / (tp + fp) 

def recall(tp, fn):
    return tp / (tp + fn) 

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) 

def calculate_metrics(data):
    tp = 0  
    fp = 0  
    fn = 0  


    for item in data:
        ground_truth = item['ground_truth']
        top_k = item['topk']
        
        found = False
        for result in top_k:
            if result['text'] == ground_truth:
                found = True
                tp += 1
            else:
                fp += 1
        
        if not found:
            fn += 1
    
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)
    f1_value = f1_score(precision_value, recall_value)
    
    return precision_value, recall_value, f1_value

def main(args):
    path = args.path
    with open(path, 'r') as f:
        search_results = json.load(f)
    precision_value, recall_value, f1_value = calculate_metrics(search_results)

    print("precision:", precision_value)
    print("recall:", recall_value)
    print("f1:", f1_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    main(args)
