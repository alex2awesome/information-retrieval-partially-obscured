import re
import json

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
            'Sources': source_dict
        })
    
    return json.dumps(parsed_articles, indent=2, ensure_ascii=False)

sources = open('sources_data_70b__200000_200100.txt', 'r')
x = sources.read()
output = parse_input(x)

with open('sources_data_70b__200000_200100.json', 'w') as f:
    f.write(output)
