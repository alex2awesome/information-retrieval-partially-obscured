import argparse
import json
import statistics

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
        ground_truth = item['ground_truth_source']
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

def dummy(data):
    count = 0
    total = 0
    for item in data:
        ground_truth = item['ground_truth_source']
        top_k = item['topk']
        
        found = False
        for result in top_k:
            if result['text'] == ground_truth:
                found = True
                count += 1
                break
        total += 1
    print("lalala", count/total)

def get_scores(path: str):
    precision_list = []
    recall_list = []
    f1_list = []

    with open(path, 'r') as file:
        contents = json.load(file)
        for content in contents:
            y_pred = set()
            y_true = set()

            y_true.add(content['ground_truth_source'])
            for k in content['topk']:
                y_pred.add(k['text'])
            
            if len(y_true) == 0:
                continue
            
            true_pos = set.intersection(y_pred, y_true)
            n = len(true_pos)

            recall = n / len(y_true)
            precision = n / len(y_pred)

            if (recall + precision) == 0:
                f1 = 0

            else:
                f1 = (2 * precision * recall) / (precision + recall)
            
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
    
    avg_prec = statistics.mean(precision_list)
    avg_rec = statistics.mean(recall_list)
    avg_f1 = statistics.mean(f1_list)

    print("average precision:", avg_prec)
    print("average recall:", avg_rec)
    print("average f1:", avg_f1)



def main(args):
    path = args.path
    with open(path, 'r') as f:
        search_results = json.load(f)
    precision_value, recall_value, f1_value = calculate_metrics(search_results)

    # print("precision:", precision_value)
    # print("recall:", recall_value)
    # print("f1:", f1_value)
    dummy(path)

    # dummy(search_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='../data_baseline/vanilla.json')
    args = parser.parse_args()
    main(args)
