import re
import json
import argparse

def parse_input(text):
    #Seperate by website_url
    articles = re.split(r'(www\.[^\s]+)', text)
    
    parsed_articles = []
    
    for i in range(1, len(articles), 2):
        article_url = articles[i].strip()
        article_content = articles[i+1].strip()
        
        sources = re.findall(r'\*\*(.*?)\*\*\n(.*?)(?=\*\*|$)', article_content, re.DOTALL)
        source_dict = {source.strip(): summary.strip() for source, summary in sources}
        
        parsed_articles.append({
            'article_url': article_url,
            'sources': source_dict
        })
    
    return json.dumps(parsed_articles, indent=2, ensure_ascii=False)


def main(args):
    filename = args.name

    path = "../../../conditional-information-retrieval/source_summaries/text_summaries/" + filename + '.txt'

    sources = open(path, 'r')
    x = sources.read()
    output = parse_input(x)

    with open(f'../data/{filename}.json', 'w') as f:
        f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    main(args)

