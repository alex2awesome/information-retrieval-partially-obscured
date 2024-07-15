import re
import json
import os
import argparse

def parse_input(text):
    #Seperate by website_url
    articles = re.split(r'(www\.[^\s]+)', text)
    
    parsed_articles = []
    
    for i in range(1, len(articles), 2):
        article_url = articles[i].strip()
        article_content = articles[i+1].strip()
        
        sources = re.findall(r'\*\*(.*?)\*\*\n(.*?)(?=\*\*|$)', article_content, re.DOTALL)
        source_dict = {source.strip(): summary.strip().strip('}') for source, summary in sources}
        
        parsed_articles.append({
            'article_url': article_url,
            'sources': source_dict
        })
    
    return json.dumps(parsed_articles, indent=2, ensure_ascii=False)

def process_file(file_path):
    basename = os.path.basename(file_path)
    if file_path.endswith('.txt') and basename not in ['requirements.txt', 'test.txt']:
        with open(file_path, 'r') as f:
            parsed_text = parse_input(f.read())
            output_name = basename.split('.')[0]
            output_path = f'../data/{output_name}.json'
            with open(output_path, 'w') as f:
                f.write(parsed_text)
                print('done!!!')

def main(args):
    # start_idx = args.start_idx
    # end_idx = args.end_idx
    # fname = f'/project/jonmay_231/spangher/Projects/conditional-information-retrieval/sources_data_70b__{start_idx}_{end_idx}.txt'
    directory = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval'
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            process_file(file_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--start_idx', type=int, default=None)
    # parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()
    main(args)

